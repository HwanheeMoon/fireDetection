from django.db import models

# Create your models here.
class detected_log(models.Model):
    id = models.AutoField(primary_key=True,auto_created=True)
    Date = models.TextField()
    Count = models.IntegerField()


    
