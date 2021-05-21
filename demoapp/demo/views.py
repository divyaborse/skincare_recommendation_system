'''from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def hi(request):
    return(render(request,'demo/hi.html'))'''
from django.shortcuts import render

import requests
import sys
import subprocess
from subprocess import run,PIPE
def button(request):
    return render(request,'demo/home.html')
def output(request):
    data=requests.get("https://www.google.com/")
    print(data.text)
    data=data.text
    return render(request,'demo/home.html',{'data':data})
def external(request):
    inp= request.POST.get('param')
    #out= run([sys.executable,'C://Users//RAJENDRA//PycharmProjects//pythonProject//demoapp//testfilepython.py',inp],shell=False,stdout=PIPE)
    #print(out)
    #return render(request,'demo/home.html',{'data1':out.stdout})
    subprocess.check_call([sys.executable, "C://Users//RAJENDRA//PycharmProjects//pythonProject//demoapp//testfilepython.py"])