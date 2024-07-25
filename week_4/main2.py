import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

path = 'images'
print("Path to images:", path)
images = []
classNames = []

# Load images from the specified directory
mylist = os.listdir(path)
print("Image files found:", mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Failed to load image: {cl}")

print("Class names:", classNames)

num_images = len(images)
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

# Display all images

for i, img in enumerate(images):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img_rgb)
    axes[i].axis('off')  # Hide axes

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


def find_encoding(images):
    encodeList = []
    for img in images:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(imgRGB)
        if encodings:
            encode = encodings[0]
            encodeList.append(encode)
        else:
            print("No face found in image.")
    return encodeList

encodeListKnown = find_encoding(images)
print('Encoding done. Number of known encodings:', len(encodeListKnown))

# Initialize webcam
# cap = cv2.VideoCapture(2) # for the webcam 
cap = cv2.VideoCapture(0) # for in pc camera

# Create a figure and axis for matplotlib
fig, ax = plt.subplots()

def update_frame(*args):
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        return

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgS = cv2.resize(imgRGB, (0, 0), None, 0.25, 0.25)

    faceCur = face_recognition.face_locations(imgS)
    encodeCur = face_recognition.face_encodings(imgS, faceCur)

    if not encodeCur:
        print("No faces detected in the current frame.")
    
    for encodeFace, faceLoc in zip(encodeCur, faceCur):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            confidence = round((1 - faceDis[matchIndex]) * 100, 2)  # Confidence score
            print(f"Detected: {name} with {confidence}% confidence")
            # y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.rectangle(img, (x1, y2 - 35), (x2, y1 - 35), (0, 255, 0), 1)
            # cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            
            # Draw polygon
            polygon_points = [
                (x1, y1), 
                (x2, y1), 
                (x2, y2), 
                (x1, y2)
            ]
            polygon_points = np.array(polygon_points, np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            cv2.polylines(img, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw label
            
            cv2.putText(img, name,(x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    

    # Convert BGR to RGB for displaying with matplotlib
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(imgRGB)
    ax.axis('off')

    plt.draw()
    plt.pause(0.001)  # Pause to update the figure

# Set up the animation and suppress the warning
ani = animation.FuncAnimation(fig, update_frame, interval=50, cache_frame_data=False)

plt.show()

cap.release()
cv2.destroy()




        # Load and convert images
# imgPri = face_recognition.load_image_file('Pri.jpg')
# imgPri = cv2.cvtColor(imgPri, cv2.COLOR_BGR2RGB)

# imgP = face_recognition.load_image_file('P.jpg')
# imgP = cv2.cvtColor(imgP, cv2.COLOR_BGR2RGB)

# Detect and draw rectangles around faces
# faceLoc = face_recognition.face_locations(imgPri)[0]
# encodePri = face_recognition.face_encodings(imgPri)[0]
# cv2.rectangle(imgPri, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# faceLocp = face_recognition.face_locations(imgP)[0]
# encodeP = face_recognition.face_encodings(imgP)[0]
# cv2.rectangle(imgP, (faceLocp[3], faceLocp[0]), (faceLocp[1], faceLocp[2]), (255, 0, 255), 2)

# # Compare faces and compute distances
# results = face_recognition.compare_faces([encodePri], encodeP)
# faceDis = face_recognition.face_distance([encodePri], encodeP)
# print("Results:", results)
# print("Face Distance:", faceDis)

# Save the images
# cv2.imwrite('P.jpg', cv2.cvtColor(imgP, cv2.COLOR_RGB2BGR))
# cv2.imwrite('Pri.jpg', cv2.cvtColor(imgPri, cv2.COLOR_RGB2BGR))
# cv2.putText(imgP , f'{results} {round(faceDis[0],2)}' ,(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

# # Display images using matplotlib
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(imgPri)
# axes[0].set_title('Pri.jpg')
# axes[0].axis('off')

# axes[1].imshow(imgP)
# axes[1].set_title('P.jpg')
# axes[1].axis('off')

# plt.show()
