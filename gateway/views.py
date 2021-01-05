from django.shortcuts import render, HttpResponse, redirect
from .hfunctions import *

# Create your views here.
def dispLogReg(request):
    return render(request, 'logReg.html')

# Registeration Submit method:
def registerSubmit(request):
    firstName = request.POST['fName']
    lastName = request.POST['lName']
    email = request.POST['eMailr']
    passWord = request.POST['pWordr']
    passWord = request.POST['chkPword']
    createHash()
    print(firstName)
    return redirect('/registered')

def success(request):
    return render(request,'success.html')

# registration methond is going to need an emailer method.