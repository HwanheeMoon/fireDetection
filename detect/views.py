from django.shortcuts import render
from .models import detected_log
from django.http import StreamingHttpResponse
from testVid import generate_frames
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

def index(request):
    return render(request, "detect/detected_list.html")

def get_data(request):
    detection_times = detected_log.objects.all()
    detect_list = []

    for t in detection_times:
        if t.Count == 0:
            detect_list.append(f"\n {t.Date} 화재가 감지됨 !!!")
        else:
            detect_list.append(f"\n {t.Date} {t.Count} 프레임 동안 감지됨")

    return HttpResponse(detect_list)

def video_stream(request,Position):
    response = StreamingHttpResponse(generate_frames(Position) ,content_type='multipart/x-mixed-replace; boundary=frame')
    if response:
        return response
    else:
        return render(request,'해당 CCTV가 연결되지 않음.')


