# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:08:24 2025

@author: ladam
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import pdfplumber
from sentence_transformers import CrossEncoder
import pandas as pd
import json
import time

#%%

if torch.cuda.is_available() : 
    print (f"La carte graphique détectée est : {torch.cuda.get_device_name(0)} \n")


#%% Chargement des pdf et chunking

# chemin absolu vers le répertoire du script
pdf_folder = os.path.join(os.path.dirname(__file__), "Documents")

print("Début du chunking \n")
# -------- 1) Préparer les documents --------
documents_text = []    # Texte classique
documents_table = []   # Tableaux
documents_list = []    # Listes / points clés

# -------- 1) Extraction PDF --------
for filename in tqdm(os.listdir(pdf_folder)):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                
                # --- 1.1 Texte classique ---
                page_text = page.extract_text() or ""
                
                # --- 1.2 Listes à puces ou numérotées ---
                # Simplification : détecte les lignes commençant par "-", "*", "•", ou chiffres
                list_lines = [line for line in page_text.splitlines() 
                              if line.strip().startswith(("-", "*", "•")) or line.strip().split(".")[0].isdigit()]
                if list_lines:
                    list_text = "\n".join(list_lines)
                    documents_list.append({
                        "text": f"[Liste, page {page_num}]\n{list_text}",
                        "metadata": {"source": filename, "page": page_num}
                    })
                
                # --- 1.3 Tableaux ---
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    table_str = "\n".join([" | ".join(cell if cell else "" for cell in row) for row in table])
                    documents_table.append({
                        "text": f"[Tableau {t_idx+1}, page {page_num}]\n{table_str}",
                        "metadata": {"source": filename, "page": page_num, "table_idx": t_idx+1}
                    })

                # --- 1.4 Texte principal (hors listes et tableaux) ---
                # Retirer les lignes déjà détectées comme listes
                filtered_text = "\n".join([line for line in page_text.splitlines() if line not in list_lines])
                if filtered_text.strip():
                    documents_text.append({
                        "text": filtered_text,
                        "metadata": {"source": filename, "page": page_num}
                    })

# -------- 2) Chunking --------
all_chunks = []
all_metadatas = []

# --- 2.1 Texte classique ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
for doc in documents_text:
    chunks = text_splitter.split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": doc["metadata"]["source"],
            "page": doc["metadata"]["page"],
            "chunk": i,
            "type": "text"
        })

# --- 2.2 Tableaux ---
table_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=32)
for tbl in documents_table:
    chunks = table_splitter.split_text(tbl["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            **tbl["metadata"],
            "chunk": i,
            "type": "table"
        })

# --- 2.3 Listes ---
list_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
for lst in documents_list:
    chunks = list_splitter.split_text(lst["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            **lst["metadata"],
            "chunk": i,
            "type": "list"
        })


# -------- 3) Création de la base vectorielle FAISS --------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectordb = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)

print ('\n Les chunks sont correctement vectorisés \n')
#%%

print ('Chargement du LLM  \n')
model_id = "vhab10/Llama-3.2-Instruct-3B-TIES"

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Charger le modèle en float16 et sur GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='cuda'  # on le gère manuellement
)

# Envoyer entièrement sur GPU
model = model.to("cuda")

# Créer le pipeline de génération
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


#%%
# Modèle de reranking
print ('\n Chargement du modèle de ranking \n')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rag_query_csv(query, k=5, initial_k=10, top_text=3, top_table=2, top_list=2, max_tokens=250):
    """
    RAG avec retrieval par type, reranking combiné, rôle spécifique (operateur | ai).
    Retourne un dictionnaire pour CSV : {"Reponse": ..., "Extraits": ..., "Temps": ...}
    """
    role = query["role"]
    question = query["request"]

    all_candidates = []

    # -------- 1) Retrieval --------
    for content_type, top_n in zip(["text", "table", "list"], [top_text, top_table, top_list]):
        candidates = vectordb.similarity_search(
            question, k=initial_k, filter={"type": content_type}
        )
        all_candidates.extend(candidates)

    if not all_candidates:
        return {"Reponse": "Je n’ai pas trouvé cette information dans les documents fournis.",
                "Extraits": "", "Temps": 0.0}

    # -------- 2) Reranking --------
    pairs = [(question, d.page_content) for d in all_candidates]
    batch_size = 8
    scores = []
    for i in range(0, len(pairs), batch_size):
        scores.extend(reranker.predict(pairs[i:i+batch_size]))

    ranked = sorted(zip(all_candidates, scores), key=lambda x: x[1], reverse=True)

    # -------- 3) Sélection finale --------
    final_docs = []
    type_counts = {"text": 0, "table": 0, "list": 0}
    type_limits = {"text": top_text, "table": top_table, "list": top_list}

    for doc, _ in ranked:
        t = doc.metadata.get("type", "text")
        if type_counts[t] < type_limits[t]:
            final_docs.append(doc)
            type_counts[t] += 1
        if sum(type_counts.values()) >= (top_text + top_table + top_list):
            break

    # -------- 4) Contexte et métadonnées --------
    context_parts = []
    excerpt_parts = []
    for d in final_docs:
        source = d.metadata.get("source", "inconnu")
        page = d.metadata.get("page", "?")
        context_parts.append(f"[{source}, page {page}] {d.page_content.strip()}")
        excerpt_parts.append(f"[{source}, page {page}]")

    context = "\n\n".join(context_parts)
    excerpts_str = "; ".join(excerpt_parts)

    # -------- 5) Prompt selon le rôle --------
    if role == "operateur":
        prompt = f""" 
Ta tâche est de répondre avec précision à la question posée en utilisant uniquement les extraits fournis et en suivant les instructions :

1. Réponds en langage naturel, clair et concis.
2. Ne réponds qu'avec les informations présentes dans les extraits.
3. Si l’information n’est pas présente, répond exactement : "Je n’ai pas trouvé cette information dans les documents fournis."
4. Ne crée pas d’informations supplémentaires.
5. Ne te répètes pas, tu peux arrêter de répondre avant la fin des tokens si tu penses que la réponse est suffisante

Extraits disponibles :
{context}

Question :
{question}

Réponse :
"""
    elif role == "ai":
        prompt = f"""
Tu es une IA et tu dois répondre au format JSON suivant :
{{
    "response_type": "list",
    "check": false,
    "calcul": -1,
    "txt": "Texte explicatif succinct basé sur les extraits.",
    "list": ["élément 1", "élément 2", "..."]
}}

Contraintes :
- Remplis "txt" avec une réponse concise en français ou 
- Remplis "list" avec les éléments pertinents trouvés dans les extrait si tu préfères répondre en liste ou 
- Remplis "calcul" si tu penses répondre avec un calcul ou 
- Remplis "check" si tu penses répondre par oui, non, vrai ou faux
- Remplis "response_type" pour indiquer le type de réponse que tu as choisi
- Si aucune information n’est trouvée, renvoie une liste vide et mets dans "txt" : "Je n’ai pas trouvé cette information dans les documents fournis."

Extraits disponibles :
{context}

Question :
{question}

Réponse :
"""
    else:
        raise ValueError("Rôle inconnu. Utiliser 'operateur' ou 'ai'.")

    # -------- 6) Appel au modèle avec mesure du temps --------
    start_time = time.time()
    result = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.0001,
        top_p=0.95
    )
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)

    output = result[0]['generated_text']
    
    if "Réponse :" in output:
        response_text = output.split("Réponse :")[-1].strip()
    else:
        response_text = output.strip()
    
    if role == "ai":
        try:
            # extraire le JSON si possible
            json_output = response_text[response_text.find("{"):response_text.rfind("}")+1]
            parsed = json.loads(json_output)
            response_text = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # fallback : renvoyer le texte brut s'il n'y a pas de JSON valide
            response_text = response_text
    
    return {"Reponse": response_text, "Extraits": excerpts_str, "Temps": elapsed}

 
# ----------------- Traitement CSV -----------------
input_csv = "Requetes.csv" 
output_csv = "Reponses.csv"

# Lire le CSV correctement
df = pd.read_csv(input_csv, encoding="utf-8")

# Liste pour stocker les résultats
results = []

# Boucle sur chaque ligne du CSV
for idx, row in df.iterrows():
    query = {"role": row["role"], "request": row["request"]}
    
    # Appel à rag_query_csv
    print (f'\n Réponse à la query : {row["request"]} \n')
    result_dict = rag_query_csv(query)
    
    # Ajouter le résultat à la liste
    results.append(result_dict)

# Créer le DataFrame de sortie
df_out = pd.DataFrame(results)

# Sauvegarder le CSV
df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n Une réponse est donnée à toutes les queries. Les résultats sont sauvegardés dans {output_csv}")
