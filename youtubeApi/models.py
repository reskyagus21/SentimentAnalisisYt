from django.db import models
import uuid

class Dokument(models.Model):
    video_url = models.URLField(max_length=500)
    comments = models.TextField()
    sentiment_results = models.TextField()
    word_frequency = models.TextField()
    csv_file = models.FileField(upload_to='csv/')
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    def __str__(self):
        return f"Dokument for {self.video_url}"

class AnalysisResult(models.Model):
    dokument = models.ForeignKey(Dokument, on_delete=models.CASCADE, related_name='analysis_results')
    pie_chart = models.ImageField(upload_to='charts/pie/', null=True, blank=True)
    bar_chart = models.ImageField(upload_to='charts/bar/', null=True, blank=True)
    analysis_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for Dokument {self.dokument.uuid}"