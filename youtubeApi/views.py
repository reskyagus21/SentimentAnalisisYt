from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
from .models import Dokument,AnalysisResult
from django.shortcuts import render
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random
import uuid
import os
import torch
import time
import csv
import re
import base64
from io import BytesIO,StringIO
from datetime import datetime
from django.core.files.base import ContentFile
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

API_KEY = ' YOUR-API_KEY '
youtube = build('youtube','v3',developerKey=API_KEY)

def komentarUrl(url):
    linkYt = urlparse(url)
    if 'youtube' in linkYt.netloc:
        return parse_qs(linkYt.query).get('v',[None])[0]
    elif 'youtube' in linkYt.netloc:
        return linkYt.path.lstrip('/')


def scrape_yt_comments(video_url, maxKomentar):
    try :
        video_id = komentarUrl(video_url)
        if not video_id:
            raise ValueError ("Url tidak ditemukan")
        komentar = []
        next_page_token = None
        while len(komentar)<=maxKomentar:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat = 'plainText'
            )
            response = request.execute()

            for item in response['items']:  
                koment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                komentar.append(koment)

                if len(komentar)>=maxKomentar:
                    break

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
        return komentar
    except Exception as e :
        logger.error(f"gagal mengambil komentar video:{str(e)}")
        return [],f"error saat mengambil video : {str(e)}"

def hapus_emote(text):
    emote = re.compile(
    "["
    "\U0001F600-\U0001F64F"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF"  
    "\U00002700-\U000027BF"  
    "\U000024C2-\U0001F251"  
    "\U0001F900-\U0001F9FF"  
    "\U0001FA70-\U0001FAFF"  
    "\U00002600-\U000026FF"  
    "\U0001F018-\U0001F270"  
    "\U0001F650-\U0001F67F"  
    "\U0001F000-\U0001F02F"  
    "]+", flags=re.UNICODE
    )
    return emote.sub(r'', str(text))

def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Netral", [0.0, 1.0, 0.0]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    sentiment = int(np.argmax(probabilities))
    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label_map[sentiment], probabilities

def tokenize_and_count(texts):
    all_words = []
    stopwords = {'ini', 'di', 'yang', 'dan', 'apa', 'siapa', 'ke', 'dia', 'saya', 'atau', 'itu', 'bisa', 'ada', 'adalah','yg','untuk'}
    for text in texts:
        words = re.findall(r'\b\w+\b', str(text).lower())
        filtered_words = [word for word in words if word not in stopwords]
        all_words.extend(filtered_words)
    word_counts = Counter(all_words)
    most_common_word = word_counts.most_common(1)[0] if word_counts else ('Tidak ada', 0)
    return word_counts, most_common_word

def create_pie_chart(sentiment_counts):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return image_png

def create_bar_chart(word_counts):
    words = list(word_counts.keys())[:10]
    counts = list(word_counts.values())[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words)
    plt.title('10 Kata Teratas')
    plt.xlabel('Frekuensi')
    plt.ylabel('Kata')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return image_png

def read_csv_file(csv_file_path):
    data = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    return data

def write_to_csv(csv_file_path, data):
    fieldnames = ['id', 'video_url', 'comments', 'sentiment_results', 'word_frequency', 'created_at']
    # Write to StringIO buffer
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    csv_content = output.getvalue()
    output.close()
    
    # Write to file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        f.write(csv_content)
    
    return csv_content.encode('utf-8')

def sentiment_analysis_view(request):
    csv_dir = os.path.join(settings.MEDIA_ROOT, 'csv')
    chart_dir = os.path.join(settings.MEDIA_ROOT,'charts')
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(os.path.join(chart_dir,'bar'), exist_ok=True)
    os.makedirs(os.path.join(chart_dir,'pie'), exist_ok=True)
    csv_file_path = os.path.join(csv_dir, 'sentiment_data.csv')
    context = {}

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'analyze':
            video_url = request.POST.get('video_url')
            if not video_url:
                context['error'] = 'URL Video TikTok diperlukan'
            else:
                # Scrape and analyze
                comments = scrape_yt_comments(video_url,500)
                if not comments:
                    context['error'] = 'Tidak ada komentar ditemukan untuk video ini'
                else:
                    cleaned_comments = [hapus_emote(comment) for comment in comments]
                    sentiments = []
                    probabilities = []
                    for comment in cleaned_comments:
                        sentiment, prob = predict_sentiment(comment)
                        sentiments.append(sentiment)
                        probabilities.append([f"{p:.2%}" for p in prob])
                    
                    sentiment_counts = Counter(sentiments)
                    word_counts, most_common_word = tokenize_and_count(cleaned_comments)

                    # Hitung jumlah sentimen
                    sentiment_summary = {
                        'Positif': sentiment_counts.get('Positif', 0),
                        'Negatif': sentiment_counts.get('Negatif', 0),
                        'Netral': sentiment_counts.get('Netral', 0)
                    }

                    # Generate UUID once
                    doc_uuid = uuid.uuid4()
                    row_id = str(doc_uuid)  # Use same UUID for CSV

                    #Simpan gambar
                    pie_chart_data = create_pie_chart(sentiment_counts)
                    bar_chart_data = create_bar_chart(word_counts)

                    # Simpan ke database
                    dokument_obj = Dokument(
                        video_url=video_url,
                        comments=json.dumps(comments),
                        sentiment_results=json.dumps(dict(sentiment_counts)),
                        word_frequency=json.dumps(dict(word_counts)),
                        uuid=doc_uuid
                    )

                    # Simpan ke CSV dan simpan path ke model
                    csv_data = read_csv_file(csv_file_path)
                    new_row = {
                        'id': row_id,
                        'video_url': video_url,
                        'comments': json.dumps(comments),
                        'sentiment_results': json.dumps(dict(sentiment_counts)),
                        'word_frequency': json.dumps(dict(word_counts)),
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    csv_data.append(new_row)
                    csv_content = write_to_csv(csv_file_path, csv_data)
                    csv_filename = f'sentiment_data_{row_id}.csv'
                    dokument_obj.csv_file.save(csv_filename, ContentFile(csv_content))

                    dokument_obj.save()

                    #Simpan ke data analysist
                    analysis_result = AnalysisResult(
                        dokument = dokument_obj,
                        analysis_data = {
                            'sentiment_counts' : dict(sentiment_counts),
                            'sentiment_summary' : sentiment_summary,
                            'word_counts' : dict(word_counts.most_common(10)),
                            'most_common_word': most_common_word,
                            'comments': comments,
                            'sentiments': sentiments,
                            'probabilities': probabilities
                        }
                    )
                    pie_filename = f'pie_chart_{row_id}.png'
                    bar_filename = f'bar_chart_{row_id}.png'
                    analysis_result.pie_chart.save(pie_filename, ContentFile(pie_chart_data))
                    analysis_result.bar_chart.save(bar_filename, ContentFile(bar_chart_data))
                    analysis_result.save()

                    # Konteks untuk template
                    context.update({
                        'comment_data': list(zip(comments, sentiments, probabilities)),
                        'sentiment_counts': dict(sentiment_counts),
                        'sentiment_summary': sentiment_summary,
                        'word_counts': dict(word_counts.most_common(10)),
                        'most_common_word': most_common_word,
                        'pie_chart': base64.b64encode(pie_chart_data).decode('utf-8'),
                        'bar_chart': base64.b64encode(bar_chart_data).decode('utf-8'),
                        'pie_chart_url': analysis_result.pie_chart.url,
                        'bar_chart_url': analysis_result.bar_chart.url,
                        'active_tab': 'results',
                        'success': 'Analisis berhasil dilakukan'
                    })

        elif action == 'view_results':
            row_id = request.POST.get('row_id')
            if not row_id:
                context['error'] = 'ID tidak ditemukan dalam permintaan'
                context['active_tab'] = 'csv'
            else:
                try:
                    uuid_obj = uuid.UUID(row_id)
                    dokument = Dokument.objects.get(uuid=row_id)
                    analysis_result = dokument.analysis_results.first()
                    
                    if not analysis_result:
                        logger.warning(f"No AnalysisResult for Dokument uuid={row_id}")
                        # Fallback: Recompute from Dokument data
                        try:
                            comments = json.loads(dokument.comments) if dokument.comments else []
                            sentiment_counts = json.loads(dokument.sentiment_results) if dokument.sentiment_results else {'Positif': 0, 'Netral': 0, 'Negatif': 0}
                            word_counts = json.loads(dokument.word_frequency) if dokument.word_frequency else {}
                            cleaned_comments = [hapus_emote(c) for c in comments]
                            sentiments = []
                            probabilities = []
                            for comment in cleaned_comments:
                                sentiment, prob = predict_sentiment(comment)
                                sentiments.append(sentiment)
                                probabilities.append([f"{p:.2%}" for p in prob])
                            sentiment_summary = {
                                'Positif': sentiment_counts.get('Positif', 0),
                                'Negatif': sentiment_counts.get('Negatif', 0),
                                'Netral': sentiment_counts.get('Netral', 0)
                            }
                            _, most_common_word = tokenize_and_count(cleaned_comments)

                            # Create new AnalysisResult
                            analysis_result = AnalysisResult(
                                dokument=dokument,
                                analysis_data={
                                    'sentiment_counts': sentiment_counts,
                                    'sentiment_summary': sentiment_summary,
                                    'word_counts': dict(Counter(word_counts).most_common(10)),
                                    'most_common_word': most_common_word,
                                    'comments': comments,
                                    'sentiments': sentiments,
                                    'probabilities': probabilities
                                }
                            )
                            pie_chart_data = create_pie_chart(sentiment_counts)
                            bar_chart_data = create_bar_chart(word_counts)
                            pie_filename = f'pie_chart_{row_id}.png'
                            bar_filename = f'bar_chart_{row_id}.png'
                            analysis_result.pie_chart.save(pie_filename, ContentFile(pie_chart_data))
                            analysis_result.bar_chart.save(bar_filename, ContentFile(bar_chart_data))
                            analysis_result.save()

                            context.update({
                                'comment_data': list(zip(comments, sentiments, probabilities)),
                                'sentiment_counts': sentiment_counts,
                                'sentiment_summary': sentiment_summary,
                                'word_counts': dict(Counter(word_counts).most_common(10)),
                                'most_common_word': most_common_word,
                                'pie_chart_url': analysis_result.pie_chart.url,
                                'bar_chart_url': analysis_result.bar_chart.url,
                                'active_tab': 'results',
                                'success': 'Hasil analisis dibuat ulang dari data Dokument'
                            })
                        except Exception as e:
                            logger.error(f"Failed to recompute AnalysisResult for uuid={row_id}: {str(e)}")
                            context['error'] = f'Gagal membuat ulang hasil analisis: {str(e)}'
                            context['active_tab'] = 'csv'
                    else:
                        analysis_data = analysis_result.analysis_data
                        comments = analysis_data.get('comments', [])
                        sentiments = analysis_data.get('sentiments', [])
                        probabilities = analysis_data.get('probabilities', [])
                        sentiment_counts = analysis_data.get('sentiment_counts', {'Positif': 0, 'Netral': 0, 'Negatif': 0})
                        word_counts = analysis_data.get('word_counts', {})
                        sentiment_summary = analysis_data.get('sentiment_summary', {'Positif': 0, 'Netral': 0, 'Negatif': 0})
                        most_common_word = analysis_data.get('most_common_word', ('Tidak ada', 0))

                        # Validasi data
                        if not isinstance(comments, list):
                            comments = []
                        if not isinstance(sentiments, list):
                            sentiments = ['Netral'] * len(comments)
                        if not isinstance(probabilities, list):
                            probabilities = [[0.0, 1.0, 0.0]] * len(comments)
                        if not isinstance(sentiment_counts, dict):
                            sentiment_counts = {'Positif': 0, 'Netral': 0, 'Negatif': 0}
                        if not isinstance(word_counts, dict):
                            word_counts = {}
                        if not isinstance(sentiment_summary, dict):
                            sentiment_summary = {'Positif': 0, 'Netral': 0, 'Negatif': 0}

                        # Load images
                        pie_chart_url = analysis_result.pie_chart.url if analysis_result.pie_chart else None
                        bar_chart_url = analysis_result.bar_chart.url if analysis_result.bar_chart else None

                        context.update({
                            'comment_data': list(zip(comments, sentiments, probabilities)),
                            'sentiment_counts': sentiment_counts,
                            'sentiment_summary': sentiment_summary,
                            'word_counts': word_counts,
                            'most_common_word': most_common_word,
                            'pie_chart_url': pie_chart_url,
                            'bar_chart_url': bar_chart_url,
                            'active_tab': 'results',
                            'success': 'Hasil analisis berhasil dimuat'
                        })
                except ValueError:
                    logger.error(f"Invalid UUID format: {row_id}")
                    context['error'] = f'ID {row_id} tidak valid'
                    context['active_tab'] = 'csv'
                except Dokument.DoesNotExist:
                    logger.error(f"Dokument not found for uuid={row_id}")
                    context['error'] = f'Data analisis tidak ditemukan untuk ID {row_id}. Mungkin sudah dihapus.'
                    context['active_tab'] = 'csv'
                except Exception as e:
                    logger.error(f"Error loading results for uuid={row_id}: {str(e)}")
                    context['error'] = f'Gagal memuat hasil analisis: {str(e)}'
                    context['active_tab'] = 'csv'


        elif action == 'update':
            row_id = request.POST.get('row_id')
            video_url = request.POST.get('video_url')
            # sentiment_results = request.POST.get('sentiment_results')
            
            csv_data = read_csv_file(csv_file_path)
            updated = False
            for row in csv_data:
                if row['id'] == row_id:
                    row['video_url'] = video_url
                    # row['sentiment_results'] = sentiment_results
                    updated = True
                    break
            if updated:
                write_to_csv(csv_file_path, csv_data)
                Dokument.objects.filter(uuid=row_id).update(video_url=video_url)
                context['success'] = 'Data berhasil diperbarui'
            else:
                context['error'] = 'Data tidak ditemukan untuk diperbarui'
            context['active_tab'] = 'csv'

        elif action == 'delete':
            row_id = request.POST.get('row_id')
            csv_data = read_csv_file(csv_file_path)
            new_csv_data = [row for row in csv_data if row['id'] != row_id]

            if len(new_csv_data) < len(csv_data):
                write_to_csv(csv_file_path, new_csv_data)

                try:
                    dokument = Dokument.objects.get(uuid=row_id)

                    # Hapus file CSV jika ada
                    if dokument.csv_file and dokument.csv_file.name:
                        csv_path = dokument.csv_file.path
                        if os.path.exists(csv_path):
                            os.remove(csv_path)

                    # Hapus bar chart dan pie chart dari AnalysisResult
                    analysis_result = dokument.analysis_results.first()
                    if analysis_result:
                        if analysis_result.pie_chart and os.path.exists(analysis_result.pie_chart.path):
                            os.remove(analysis_result.pie_chart.path)
                        if analysis_result.bar_chart and os.path.exists(analysis_result.bar_chart.path):
                            os.remove(analysis_result.bar_chart.path)

                    # Hapus dari database
                    dokument.delete()

                    context['success'] = 'Data berhasil dihapus'
                except Dokument.DoesNotExist:
                    context['error'] = f'Data dengan ID {row_id} tidak ditemukan'
                except Exception as e:
                    logger.error(f"Gagal menghapus data: {str(e)}")
                    context['error'] = f'Terjadi kesalahan saat menghapus data: {str(e)}'
            else:
                context['error'] = 'Data tidak ditemukan untuk dihapus'

            context['active_tab'] = 'csv'

    context['csv_data'] = read_csv_file(csv_file_path)
    return render(request, './index.html', context)
