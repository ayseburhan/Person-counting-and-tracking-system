import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import socket
import datetime
from sort import *

def firebase_kaydet():
    if not firebase_admin._apps:
        cred = credentials.Certificate('keys.json')
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    kullanici_adi = input("Kullanıcı Adınızı Girin: ")
    simdiki_tarih_saat = datetime.datetime.now()

    # Burada tracker ve detections oluşturulmalı ve görüntü işleme adımları yapılmalıdır
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    detections = np.empty((0, 5))
    resultsTracker = tracker.update(detections)
    kimlikler = [] 

    for result in resultsTracker:
        Id = result.astype(int)
        kimlikler.append([Id])  # Her bir Id'yi listeye ekleyin

    doc_ref = db.collection(u'kullanicilar').document()
    bilgisayar_ip = socket.gethostbyname(socket.gethostname())

    doc_ref.set({
        u'kullanici_adi': kullanici_adi,
        u'tarih': simdiki_tarih_saat.strftime("%Y-%m-%d"),
        u'saat': simdiki_tarih_saat.strftime("%H:%M:%S"),
        u'ip_adresi': bilgisayar_ip,
        u'kimlik': kimlikler
    })

    print("Veri Firebase'e başarıyla kaydedildi!")
