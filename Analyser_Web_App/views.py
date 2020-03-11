from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Create your views here.


def getfile(request):

    return predict_result(request)


def predict_result(request):

    return render(request, 'html/predict_result.html')


def upload_data_sets(request):

    return render(request, 'html/upload_data_sets.html')


def save_train_data(request):

    if request.method == "POST":
        upload_file = request.FILES['train_data']
        fs = FileSystemStorage()
        fs = fs.location+"/Train_Data_Set"
        fs = FileSystemStorage(fs)
        fs.save(upload_file.name, upload_file)
    return render(request, 'html/upload_train_data.html')


def save_test_data(request):

    if request.method == "POST":
        upload_file = request.FILES['test_data']
        fs = FileSystemStorage()
        fs = fs.location+"/Test_Data_Set"
        fs = FileSystemStorage(fs)
        fs.save(upload_file.name, upload_file)
    return render(request, 'html/upload_test_data.html')


