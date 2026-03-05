import numpy as np
import cv2 as cv

rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:

        # ========= CARA =========
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        # ========= OREJAS =========
        oreja_y = y + int(h*0.5)

        cv.circle(img,(x-10,oreja_y),20,(200,180,160),-1)
        cv.circle(img,(x+w+10,oreja_y),20,(200,180,160),-1)

        # ========= OJOS =========
        ojo_y = y + int(h*0.35)

        ojo1 = (x + int(w*0.3), ojo_y)
        ojo2 = (x + int(w*0.7), ojo_y)

        cv.circle(img,ojo1,20,(255,255,255),-1)
        cv.circle(img,ojo2,20,(255,255,255),-1)

        cv.circle(img,ojo1,7,(0,0,0),-1)
        cv.circle(img,ojo2,7,(0,0,0),-1)

        # ========= NARIZ =========
        nariz = (x + w//2 , y + int(h*0.55))
        cv.circle(img,nariz,8,(0,0,0),-1)

        # ========= BIGOTE =========
        bigote_y = y + int(h*0.65)

        cv.ellipse(img,(x+int(w*0.45),bigote_y),(25,10),0,0,180,(0,0,0),4)
        cv.ellipse(img,(x+int(w*0.55),bigote_y),(25,10),0,0,180,(0,0,0),4)

        # ========= BOCA =========
        boca = (x + w//2 , y + int(h*0.8))
        cv.ellipse(img,boca,(30,15),0,0,180,(0,0,255),3)

        # ========= SOMBRERO MARIACHI GRANDE =========

        centro_sombrero = (x + w//2 , y - int(h*0.2))

        # ALA GRANDE
        cv.ellipse(
            img,
            centro_sombrero,
            (int(w*0.9), int(h*0.25)),
            0,
            0,
            360,
            (0,0,0),
            -1
        )

        # BORDE DECORATIVO
        cv.ellipse(
            img,
            centro_sombrero,
            (int(w*0.9), int(h*0.25)),
            0,
            0,
            360,
            (0,215,255),
            4
        )

        # COPA DEL SOMBRERO
        top1 = (x + int(w*0.25), y - int(h*0.2))
        top2 = (x + int(w*0.75), y - int(h*0.2))
        top3 = (x + int(w*0.65), y - int(h*0.6))
        top4 = (x + int(w*0.35), y - int(h*0.6))

        pts = np.array([top1,top2,top3,top4], np.int32)

        cv.fillPoly(img,[pts],(0,0,0))

        # DECORACIÓN SOMBRERO
        cv.line(img,top4,top3,(0,215,255),4)

    cv.imshow("Mariachi Face", img)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()