from django.db import models

# Create your models here.

class Keyword(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    hash = models.CharField(max_length=64)
    
