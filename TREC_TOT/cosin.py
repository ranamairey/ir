import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ir_datasets
from prossing import TextPreprocessor
import json

# تعريف دالة لحساب Precision@K
def precision_at_k(y_true, y_pred, k):
    if k <= 0:
        raise ValueError("k يجب أن يكون عددًا صحيحًا موجبًا")
    if len(y_true) < k or len(y_pred) < k:
        raise ValueError("طول y_true و y_pred يجب أن يكون على الأقل k")
    
    true_positives = sum(y_pred[:k])
    return true_positives / k

# تعريف دالة لحساب Recall@K
def recall_at_k(y_true, y_pred, k):
    if k <= 0:
        raise ValueError("k يجب أن يكون عددًا صحيحًا موجبًا")
    if len(y_true) < k or len(y_pred) < k:
        raise ValueError("طول y_true و y_pred يجب أن يكون على الأقل k")
    
    relevant_docs = sum(y_true)
    true_positives = sum(y_pred[:k])
    return true_positives / relevant_docs if relevant_docs > 0 else 0

# تعريف دالة لحساب MAP
def mean_average_precision(y_true_all, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    num_relevant_docs = sum(y_true_all)
    average_precision = 0.0
    num_retrieved_relevant_docs = 0
    for i, idx in enumerate(sorted_indices):
        if y_true_all[idx]:
            num_retrieved_relevant_docs += 1
            precision_at_i = num_retrieved_relevant_docs / (i + 1)
            average_precision += precision_at_i
    if num_relevant_docs > 0:
        average_precision /= num_relevant_docs
    return average_precision

# تعريف دالة لحساب MRR
def reciprocal_rank(y_true, y_pred):
    for i, val in enumerate(y_pred):
        if val:
            return 1 / (i + 1)
    return 0

# تهيئة معالج النصوص
text_processor = TextPreprocessor()

# تحميل مجموعة بيانات TREC
dataset = ir_datasets.load("trec-tot/2023/train")

# تحميل النصوص من ملف TSV
corpus_file = r"C:\Users\Dell X360-Gen8\Desktop\preprocessed_data5.tsv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['doc_id', 'text'])

# حذف الصفوف التي تحتوي على قيم NaN في عمود النص (إذا وجدت)
df = df.dropna(subset=['text'])

# إنشاء قائمة من الوثائق
documents = df['text'].tolist()  # اختيار جميع الوثائق
doc_ids = df['doc_id'].tolist()

# تحويل النصوص باستخدام TF-IDF
vectorizer = TfidfVectorizer(
    min_df=10,  # تجاهل المصطلحات التي تظهر في أقل من 10 وثائق
    max_df=0.8,  # تجاهل المصطلحات التي تظهر في أكثر من 80% من الوثائق
    ngram_range=(1, 2),  # أخذ كل من الأحادية والثنائية
    smooth_idf=True,  # تطبيق التنعيم على أوزان IDF
    sublinear_tf=True  ,# تطبيق قياس TF تحت الخطي
)
tfidf_matrix = vectorizer.fit_transform(documents)

# تحميل qrels وتحويلها إلى DataFrame
qrels = list(dataset.qrels_iter())
qrels_df = pd.DataFrame(qrels)

# تحميل الاستعلامات من ملف JSONL
queries_file = r"C:\Users\Dell X360-Gen8\.ir_datasets\trec-tot\2023\train\queries.jsonl"
with open(queries_file, 'r') as f:
    queries = [json.loads(line) for line in f]

# قوائم لتخزين الدقة والاستدعاء ومتوسط الدقة لكل استعلام
average_precisions = []
precisions = []
recalls = []
reciprocal_ranks = []


top_n = 10


for query in queries:
    query_id = query['id']
    query_text = query['text']

    # معالجة الاستعلام
    processed_query = text_processor.preprocess_query(query_text)

    # تحويل الاستعلام إلى متجه TF-IDF باستخدام المحول المحمل مسبقًا
    query_tfidf_vector = vectorizer.transform([processed_query])

    # حساب تشابه الكوسينوس بين الاستعلام وجميع الوثائق
    query_cosine_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()

    # الحصول على الأحكام ذات الصلة للاستعلام الحالي
    relevant_docs = qrels_df[(qrels_df['query_id'] == query_id) & (qrels_df['relevance'] > 0)]['doc_id'].tolist()

    # فرز الوثائق بناءً على درجات التشابه بترتيب تنازلي
    sorted_doc_indices = np.argsort(query_cosine_similarities)[::-1]

    # الحصول على معرفات الوثائق ذات الصلة
    relevant_doc_ids = [doc_ids[idx] for idx in sorted_doc_indices]

    # الحصول على الأحكام ذات الصلة لأفضل الوثائق
    y_true = np.array([doc_id in relevant_docs for doc_id in relevant_doc_ids[:top_n]])  # تسميات الصلة الحقيقية
    y_pred = np.array([1 if doc_id in relevant_docs else 0 for doc_id in relevant_doc_ids[:top_n]])  # تسميات الصلة المتوقعة

    y_true_all = np.array([doc_id in relevant_docs for doc_id in doc_ids])
    y_scores = query_cosine_similarities
    average_precision = mean_average_precision(y_true_all, y_scores)
    average_precisions.append(average_precision)
        # احسب الدقة والاستدعاء للاستعلام
    precision = precision_at_k(y_true, y_pred, top_n)
    recall = recall_at_k(y_true, y_pred, top_n)

        # أضف الدقة والاستدعاء إلى قوائمهم الخاصة
    precisions.append(precision)
    recalls.append(recall)

    # احسب MRR للاستعلام
    reciprocal_rank_val = reciprocal_rank(y_true, y_pred)
    reciprocal_ranks.append(reciprocal_rank_val)

# احسب متوسط الدقة والاستدعاء
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)

# احسب متوسط متوسط الدقة (MAP)
mean_average_precision = np.mean(average_precisions)

# احسب MRR
mean_reciprocal_rank = np.mean(reciprocal_ranks)

print(f"mean_precision {mean_precision}")
print(f"mean_recall {mean_recall}")
print(f"(MAP): {mean_average_precision}")
print(f"(MRR): {mean_reciprocal_rank}")
