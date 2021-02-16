from django.db import models

# Create your models here.

class StepCount_Data(models.Model):
    saved_time = models.DateField(primary_key=True)
    stepCount = models.IntegerField()

    def __str__(self):
        return self.saved_time

    class Meta:
        db_table = 'stepcountData'