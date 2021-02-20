from django.db import models

# Create your models here.
class lotto(models.Model):
    Draw = models.SmallIntegerField(primary_key=True, null=False) 
    Ball_1 = models.FloatField(null=False)
    Ball_2 = models.FloatField(null=False)
    Ball_3 = models.FloatField(null=False)
    Ball_4 = models.FloatField(null=False)
    Ball_5 = models.FloatField(null=False)
    Ball_6 = models.FloatField(null=False)
    Bonus = models.FloatField(null=False)
    Powerball = models.FloatField(null=False)
    Sorted = models.BinaryField(null=False)
    Multivariable = models.BinaryField(null=False)
    Version = models.SmallIntegerField(null=True)
