import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

# Firebase'e bağlanmak için servis hesabı anahtar dosyasının yolunu belirtin
cred = credentials.Certificate('keys.json')  # Servis hesabı anahtar dosyasının yolunu verin
firebase_admin.initialize_app(cred)

# Firestore veritabanına erişim sağlayın
db = firestore.client()

# Kullanıcıdan kullanıcı adını alın
kullanici_adi = input("Kullanıcı Adınızı Girin: ")

# Tarih ve saati alın
simdiki_tarih_saat = datetime.datetime.now()

# Firestore'a veri ekleyin
doc_ref = db.collection(u'kullanicilar').document()
doc_ref.set({
    u'kullanici_adi': kullanici_adi,
    u'tarih': simdiki_tarih_saat.strftime("%Y-%m-%d"),
    u'saat': simdiki_tarih_saat.strftime("%H:%M:%S")
})

print("Veri Firebase'e başarıyla kaydedildi!")
