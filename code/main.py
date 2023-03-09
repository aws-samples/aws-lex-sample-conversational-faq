import torch
import os
import json
import boto3
import string
import transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
# from nltk import tokenize

from utils import str2bool, normalize_text, split_into_sentences


INSTRUCTION_CQA = "Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge."
INSTRUCTION_GR = "Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge."
GENERATOR_MODEL_PATH = "microsoft/GODEL-v1_1-base-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_PATH)

EMBEDDING_MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)


FAQ_INTENT_NAME = "FallbackIntent"
UNANSWERABLE_TAG = "CANNOTANSWER"
FALLBACK_ANSWER = "[OOD] No relevant information found."
OOD_TAG = "[OOD]"
TOP_K = 20  # reranking top
TOP_N_GEN = 3


# ==== OpenSearch: Uncomment below for OpenSearch Index Configurations ====
# DOC_MAX_LEN = 145  # 145 for 3 and 80 for 5
# QUERY_MAX_LEN = 50
# host = "YOUR_HOST"  # opensearch endpoint
# region = "us-east-1"
# credentials = boto3.Session().get_credentials()
# auth = AWSV4SignerAuth(credentials, region)
# client = OpenSearch(
#     hosts=[{"host": host, "port": 443}],
#     http_auth=auth,
#     use_ssl=True,
#     verify_certs=True,
#     connection_class=RequestsHttpConnection,
# )


# ==== Kendra: Uncomment below for Kendra Index Configurations ====
client = boto3.client("kendra")
DOC_MAX_LEN = 80  # 145 for 3 and 80 for 5
QUERY_MAX_LEN = 50


def post_processing_godel(input_str, passages=[]):
    """
    Post-process the generative answer and filter out the ungrounded one.
    """
    input_str = input_str.strip()
    answer = input_str
    if input_str.endswith("?"):
        return OOD_TAG + answer
    if input_str.startswith(UNANSWERABLE_TAG):
        if (
            input_str.endswith("?")
            or input_str.endswith("###")
            or input_str.endswith(UNANSWERABLE_TAG)
        ):
            return OOD_TAG + answer
        if "/" in input_str:
            answer = input_str.split("/", 1)[-1].replace("/", "")
        if "?" in input_str:
            answer = input_str.split("?", 1)[-1]
    if "?" in input_str:
        answer = input_str.split("?", 1)[-1]
    if passages:
        passages_all = normalize_text(" ".join(passages))
        answer = normalize_text(answer)
        if answer.lower() in passages_all.lower():
            return answer
        # sentences = tokenize.sent_tokenize(answer)
        sentences = split_into_sentences(answer)
        if len(sentences) > 1 and sentences[1].lower() in passages_all.lower():
            return " ".join(sentences[1:])
        return OOD_TAG + answer
    answer = answer.strip()
    if not answer:
        return FALLBACK_ANSWER
    return answer


def get_reranking_results(
    query_str,
    contexts,
    mode="qa",
    is_instructed=False,
    query_instruct="",
    doc_instruct=""
):
    """
    Rerank the top K passage based on semantic similarities between the query and doc embeddings.
    """
    sentences_doc = []
    lst_pid = []
    for i, ex in enumerate(contexts):
        lst_pid.append(ex["pid"])
        text = " ".join(ex["text"].split()[:DOC_MAX_LEN])
        if mode == "query":
            doc_text = ex["title"]
        elif mode == "answer":
            doc_text = text
        else:
            doc_text = ex["title"] + " " + text
        if is_instructed:
            sentences_doc.append([doc_instruct, doc_text, 0])
        else:
            sentences_doc.append(doc_text)
    if is_instructed:
        sentences_query = [[query_instruct, query_str, 0]]
    else:
        sentences_query = [query_str]
    embeddings_query = embedding_model.encode(sentences_query)
    embeddings_doc = embedding_model.encode(sentences_doc)
    similarities = cosine_similarity(embeddings_query, embeddings_doc)
    array = zip(range(len(similarities[0])), similarities[0])
    array_sorted = sorted(array, key=lambda x: x[1], reverse=True)
    contexts_rerank = []
    for idx, score in array_sorted:
        contexts[idx]["score_rerank"] = str(round(score, 4))
        contexts_rerank.append(contexts[idx])
    return contexts_rerank


def get_conversational_query(input_str, dial_hist_uttr, separator="[SEP]"):
    """
    Construct query based on current turn and history.
    """
    input_str = input_str.strip()
    turns = dial_hist_uttr + [input_str]
    if input_str[-1] not in string.punctuation:
        input_str += "."
    query_str = input_str
    if dial_hist_uttr:
        query_str = input_str + separator + " || ".join(dial_hist_uttr[::-1])
    query_str = " ".join(query_str.split()[:QUERY_MAX_LEN])
    return query_str, turns[::-1]


def get_retrievals_opensearch(query, client, index_name, max_k=10):
    """
    Get retrieval results from OpenSearch index.
    """
    query = query.replace("[SEP]", "")
    q = {
        "size": max_k,
        "query": {"multi_match": {"query": query, "fields": ["title", "content"]}},
    }
    response = client.search(body=q, index=index_name)
    contexts = []
    for ele in response["hits"]["hits"]:
        contexts.append(
            {
                "pid": ele["_id"],
                "title": ele["_source"]["title"],
                "text": ele["_source"]["content"],
                "score": ele["_score"],
            }
        )
    id_ = ""
    return {
        "id": id_,
        "question": query,
        "contexts": contexts,
    }


def get_retrievals_kendra(query, index_id, d_id_doc=None):
    """
    Get retrieval results from Kendra index.
    """
    id_ = ""
    contexts = []
    kendra_a_text = "NA"  # query result type - "ANSWER"
    kendra_qa_text = "NA"  # query result type - "QUESTION_ANSWER"
    response = client.query(
        QueryText=query,
        IndexId=index_id,
        QueryResultTypeFilter="DOCUMENT",
        PageSize=50,
    )
    for query_result in response["ResultItems"]:
        if query_result["Type"] == "ANSWER":
            kendra_a_text = query_result["DocumentExcerpt"]["Text"]
        if query_result["Type"] == "QUESTION_ANSWER":
            kendra_qa_text = query_result["DocumentExcerpt"]["Text"]
        if query_result["Type"] == "DOCUMENT":
            title = ""
            if "DocumentTitle" in query_result:
                title = query_result["DocumentTitle"]["Text"]
            doc_id = query_result["DocumentId"]
            if d_id_doc and doc_id in d_id_doc and "contexts" in d_id_doc[doc_id]:
                document_text = d_id_doc[doc_id]["contexts"]
            else:
                document_text = normalize_text(
                    query_result["DocumentExcerpt"]["Text"][3:-3]
                )
            score = query_result["ScoreAttributes"]["ScoreConfidence"]
            contexts.append(
                {"pid": doc_id, "title": title, "text": document_text, "score": score}
            )
    return {
        "id": id_,
        "question": query,
        "contexts": contexts,
        "kendra_a_text": kendra_a_text,
        "kendra_qa_text": kendra_qa_text,
    }


def generate(turns, contexts, model_name="godel"):
    """
    Generate answer text given query and knowledge contexts.
    """
    lst = []
    for ex in contexts:
        c = ex["title"].split("###")[0] + " " + ex["text"]
        lst.append(" ".join(c.split()[:120]))
    knowledge = " || ".join(lst)
    if model_name == "godel":
        dialog = " EOS ".join(turns)  # turns from oldest to latest
        dialog = " ".join(dialog.split()[(0 - QUERY_MAX_LEN) :])
        query = f"{INSTRUCTION_CQA} [CONTEXT] {dialog} [KNOWLEDGE] {knowledge}"
    else:
        dialog = turns[-1] + " [SEP] " + " ".join(turns[:-1])
        dialog = " ".join(dialog.split()[:QUERY_MAX_LEN])
        query = f"{dialog} [KNOWLEDGE] {knowledge}"
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids, max_length=50, min_length=6, top_p=0.9, do_sample=False
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


def process_dial_hist(dial_hist):
    """
    Process Lex v2 session attributes for dialogue utterances.
    """
    dial_hist_uttr = []
    dial_act_pre = None
    for ex in dial_hist["history"]["L"]:
        ex = ex["M"]
        utterance = "user: " + ex["user"]["S"].strip()
        response = ex["user"]["S"].strip()
        intent_name = ex["intent"]["S"]
        slot_values = [k for k, v in ex["slot_values"]["M"].items() if v["S"]]
        # is_include = (not dial_act_pre) or (not slot_values) or (intent_name.lower() in ["fallbackintent", "faq"])
        is_include = intent_name.lower() in ["fallbackintent", "faq"]
        if response and response.startswith(OOD_TAG):
            is_include = False
            dial_act_pre = OOD_TAG
        else:
            dial_act_pre = None
        if is_include:
            dial_hist_uttr.append(utterance)
        dial_act_pre = ex["dialogAction"]["M"].get("type", {"S": ""}).get("S")
    return dial_hist_uttr, dial_act_pre


def cfaq_api(payload):
    """
    Main CFAQ API.
    """
    data = json.loads(payload)
    response = FALLBACK_ANSWER

    is_rerank = str2bool(data.get("is_rerank", True))
    is_single = str2bool(data.get("is_single", True))
    index_name = data["index_id"]
    service_type = data.get("index_type", "kendra")
    query_str = data["query"]  # single turn for now
    if service_type == "opensearch":
        retrieval_results = get_retrievals_opensearch(query_str, client, index_name)
    else:
        retrieval_results = get_retrievals_kendra(query_str, index_name)
    if not retrieval_results.get("contexts", []):
        return {"text": response, "contexts": [], "OOD": True}
    contexts = retrieval_results["contexts"]
    score = contexts[0]["score"]
    if score == "VERY_HIGH":
        contexts = contexts[:1]
    elif is_rerank:
        contexts = get_reranking_results(query_str, retrieval_results["contexts"][:TOP_K])
    if is_single:
        query_input = [query_str]
    else:
        dial_hist_uttr, da_pre = process_dial_hist(
            data.get("dial_hist_lex", {"history": {"L": {}}})
        )
        _, turns = get_conversational_query(query_str, dial_hist_uttr)
        query_input = turns
    contexts = contexts[:TOP_N_GEN]
    passages_cfaq = [ele["text"] for ele in contexts]
    passage_cfaq = passages_cfaq[0]
    passage_search = retrieval_results["contexts"][0]["text"]
    if passage_search not in passages_cfaq:
        contexts.append(retrieval_results["contexts"][0])
    response = generate(query_input, contexts, model_name="godel")
    response_post = post_processing_godel(response, passages_cfaq)
    is_ood = response == FALLBACK_ANSWER
    return {
        "text": f"CFAQ-ANSWER: {response_post} #### CFAQ-PASSAGE: {passage_cfaq}",
        "contexts": contexts,
        "OOD": is_ood,
    }


def model_fn(model_dir):
    """
    Load the model for inference
    """
    model_dict = {"model": model, "tokenizer": tokenizer}
    return model_dict


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    response = cfaq_api(input_data)
    return response


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    if response_content_type == "application/json":
        response = prediction
    else:
        response = str(prediction)

    return response