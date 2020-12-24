from django.shortcuts import render, HttpResponse, redirect

# Create your views here.
def dispLogReg(request):
    return render(request, 'logReg.html')

# Registeration Submit method:
def registerSubmit(request):
    return redirect('/registered')

def success(request):
    return render(request,'success.html')

# registration methond is going to need an emailer method.