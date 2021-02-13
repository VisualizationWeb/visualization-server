from django.contrib import admin
from .models import StepCount_Data

# Register your models here.

# 관리자 페이지 보기 방식
class StepCount_DataAdmin(admin.ModelAdmin):
    list_display = ('saved_time', 'stepCount')

admin.site.register(StepCount_Data, StepCount_DataAdmin)