from rest_framework import serializers
from .serializers import *
from rest_framework.response import Response
from .models import *


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('name','start_date','end_date','comments','status')

