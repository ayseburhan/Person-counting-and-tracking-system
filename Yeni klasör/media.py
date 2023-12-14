import mysql.connector

# MySQL bağlantısı
mydb = mysql.connector.connect(
    user=  "root", 
    host= "localhost" ,
    password= "root123",
    port= "3306",
    database="project"
)

mycursor = mydb.cursor()

# Veritabanında tablo oluşturma
mycursor.execute("CREATE TABLE IF NOT EXISTS PeopleCount (ID INT AUTO_INCREMENT PRIMARY KEY, PersonID INT)")

# Fonksiyon veritabanına ID eklemek için
def add_person_to_db(person_id):
    sql = "INSERT INTO PeopleCount (PersonID) VALUES (%s)"
    val = (person_id,)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "record inserted.")

# Toplam ID sayısını alma
def get_total_count():
    mycursor.execute("SELECT COUNT(*) FROM PeopleCount")
    result = mycursor.fetchone()
    return result[0]
