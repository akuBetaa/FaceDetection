#mengimpor library opencv
import cv2


#membuat objek pendeteksi bernama 'face_ref', 'face_ref.xml' berisi model deteksi wajah
face_ref = cv2.CascadeClassifier("face_ref.xml")
#membuka kamera default komputer/laptop
camera = cv2.VideoCapture(0)


#mendefinisikan fungsi bernama "face_detection" yg menerima frame gambar sebagai input
def face_detection(frame):
    #mengkonversi gambar dari format warna (RGB) menjadi grayscale [deteksi wajah berfungsi baik dengan gambar grayscale]
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #menggunakan 'face_ref' untuk mendeteksi wajah dari gambar yg dikonversi grayscale
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1)
    #mengembalikan hasil deteksi wajah
    return faces

#mendefinisikan fungsi bernama "frame"
def drawer_box(frame):
    #untuk memulai loop untuk menjalankan deteksi wajah
    for x, y, w, h in face_detection(frame):
        #untuk menggambar sebuah kotak persegi panjang disekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x +w, y+h), (255, 255, 255), 4)
        #untuk menuliskan teks "Beta NA" dibawah kotak wajah yang terdeteksi
        cv2.putText(frame, "Beta NA", (x , y + h + 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


#mendefinisikan fungsi untuk menutup jendela tampilan kamera
def close_window():
    #
    camera.release()
    #menutup semua jendela kamera
    cv2.destroyAllWindows()
    #keluar dari program python
    exit()


#berisi loop utama program
def main():
    while True:
        #membaca frame gambar dari kamera disimpan ke variabel 'frame'
        _, frame = camera.read()
        #memanggil fungsi "drawer_box" untuk menggambar kotak dan text
        drawer_box(frame)
        #menampilkan frame gambar dengan kotak dan teks berjudul "Beta_FaceDetection"
        cv2.imshow("Beta_FaceDetection", frame)

        #untuk memeriksa inputan user pada tombol 'x' akan menutup jendela
        if cv2.waitKey(1) & 0xFF == ord('x'):
            #berfungsi menutup jendela
            close_window()


#untuk memastikan kode dijalakan ketika python dijalankan berlangsung
if __name__ == '__main__':
    #memanggil fungsi main untuk memulai loop utama program
    main()