from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from . python_functions.functions import pubmedAbstractDownload, getAbstractFromEntrezQuery, getCandidates, getCandidatePredictions

def home(request):
    context = {
        'title': 'Data Extraction Demo'
    }
    return render(request, 'abstract_demo_app/home.html', context)

# AJAX PAGES
def getArticle(request):
    id = request.POST["artID"]
    abstract = pubmedAbstractDownload(id)
    abstract = getCandidates(abstract) #preprocess
    abstractDict = {'text': abstract}
    return render(request, 'abstract_demo_app/getArticle.html', abstractDict)

def getArticlePredictions(request):
    text = request.POST["abstract_text"]
    candidateInfo = getCandidatePredictions(text)
    return render(request, "abstract_demo_app/getArticlePredictions.html", {'text': candidateInfo})

