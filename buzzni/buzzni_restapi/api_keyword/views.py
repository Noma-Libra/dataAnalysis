from rest_framework.views import APIView
from rest_framework.response import Response

class KeywordView(APIView):
    def get(self, request):
        return Response('OK', status=200)

# Create your views here.
