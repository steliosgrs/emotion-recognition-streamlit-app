import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import StringIO
import streamlit as st
import tensorflow
from tensorflow import keras
import time
import threading as th
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode, ClientSettings
import av
import copy
import random
from PIL import Image
import os
from utils import *
# from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from streamlit.scriptrunner import add_script_run_ctx as add_report_ctx

st.set_page_config(layout="wide")

file = 'emotions.txt'
# Load model
emotion_dict = {0:'angry', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'neutral', 5:'sad' , 6:'surprise'}
classifier = load_model('cnn_model200_32.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
Current_emotion = []
# thread = th.Thread(target=timer,daemon=True)
# add_report_ctx(thread)
# t = thread.start()
last5frames = []
lock = th.Lock()
emotions_list=[]


# Load face
try:
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Cascade for face detection loaded")
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


def most_frequent(List):
    counter = 0
    emotion = List[0]

    for em in List:
        curr_frequency = List.count(em)
        if (curr_frequency / 5) > 0.5:
            emotion = em
            break
    #         if(curr_frequency> counter):
    #             counter = curr_frequency
    #             emotion = em

    return emotion

st.set_option('deprecation.showPyplotGlobalUse', False)
def emotion_analysis(emotions):

    # objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    objects = ('angry', 'disgust', 'fear', 'happy','neutral', 'sad', 'surprise' )
    y_pos = np.arange(len(objects))
    fig = plt.figure()
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    # st.pyplot(fig)

    # fig = plt.figure()
    # fig = plt.bar(y_pos, emotions, align='center', alpha=0.5)
    # fig = plt.xticks(y_pos, objects)
    # fig = plt.ylabel('percentage')
    # fig = plt.title('emotion')

    # plot_em = plt.show()
    # st.pyplot(plot_em)
    return fig


def facecrop(image,name):

    image = image[::-1]
    index = image.find('\\')
    image = image[::-1]
    dir_name = image[:-index]

    # print(dir_name)

    frame = cv2.imread(image)
    # frame = cv2.cvtCoLOR(image, cv2.IMREAD_COLOR)
    # image = image.to_ndarray(format="bgr24")

    # Load the cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    face_roi = cv2.imread(image)
    # img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img = cv2.imread(image,cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(img,1.1,4)
    for x,y,w,h in faces:
        roi_gray = img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y), (x+w , y+h), (255,0,0), 2)
        facess = cascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print('Face not detected\n')
        else:
            for (ex, ey, ew, eh) in facess:
                # Crop the face
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]

                # os.path.join(dir_name, 'cropped')
                dir =os.path.join('.', f"Cropped {name}")
                # dir = os.path.join(dir_name, f"Cropped {name}")
                # print(dir)
                # print(face_roi)
                # cv2.imwrite(dir, face_roi)
                cv2.imwrite(dir, face_roi)
                # print(f"auto einai to dir {dir}")
                # time.sleep(1)
                # print(type(face_roi))


    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Print original image with face detection box
    st.image(frame)
    return face_roi

def image_classification(image):


    # Take the name of the image
    image = image[::-1]
    index = image.find('\\')
    image = image[::-1]
    name = image[-index:]

    # print(f"NAME {name}")

    cropped = facecrop(image,name)
    # print(f"auto einai cropped {cropped}")

    # full_path = os.path.join(dir,image)

    res_path = 'Cropped ' + name
    # res_path = os.path.join('Cropped ', name) # δημιουργεί bug

    # full_path2 = os.path.join(dir,res_path)
    # print(f"RES PATH {res_path}")

    img = keras.preprocessing.image.load_img(res_path, target_size=(48, 48), color_mode="grayscale")
    # os.remove(full_path2)

    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = classifier.predict(x)


    plothem = emotion_analysis(custom[0])

    prediction = classifier.predict(x)[0]
    maxindex = int(np.argmax(prediction))
    finalout = emotion_dict[maxindex]
    output = str(finalout)
    st.write(f"The face in the image looking {output}")
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    return output, plothem


class FaceEmotion(VideoProcessorBase):


    def Face_Emotion(self, frame):
        output=""
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        labels = []
        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h),
                          color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                # print(label)

                if len(last5frames) < 5:
                    last5frames.append(label)  # New
                elif len(last5frames) == 5:
                    print(last5frames)
                    count = last5frames
                    last5frames.pop(0)
                else:
                    last5frames.pop(0)  # New

                Current_emotion = most_frequent(last5frames)
                with lock:
                    emotions_list.append(Current_emotion)

                # choices(Current_emotion)
                # if t == 0:
                #     choices(Current_emotion)
                #     t = 5

                # Print the percent and the % in position
                percent = int(prediction[prediction.argmax()] * 100)  # New
                percent = str(percent) + '%'
                percent_position = (x + w, y - 10)  # New

                # Get Current dominant emotion
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(percent), percent_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # New

        return img, output

    def recv(self, frame):
        # img = frame.to_ndarray(format="bgr24")
        # img = cv2.flip(img, 1)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # image = copy.deepcopy(img)

        img, result = self.Face_Emotion(frame)

        if result == "":
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            self.current_emotion = result[0]
            return  av.VideoFrame.from_ndarray(img, format="bgr24")

# def dir_selector(folder_path='.'):
#     dirnames = os.listdir(folder_path)
#     # selected_filename = st.selectbox('Select a file', filenames)
#     selected_dirname = st.selectbox('Ο φάκελος ', dirnames)
#     st.write('Ο φάκελος πρέπει να είναι της μορφής: User\\Pictures\\ ή C:\\Users\\User\\Pictures\\')
#     return os.path.join(folder_path, selected_dirname)

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    # selected_filename = st.selectbox('Select a file', filenames)
    selected_filename = st.selectbox('Το αρχείο (.jpg, .png)', filenames)
    return os.path.join(folder_path, selected_filename)


def main():

    st.title("Αυτόματη Ανίχνευση, Ανάλυση και Αναγνώριση Συναισθημάτων")
    # activiteis = ["Home", "Analyze Image Emotion", "Webcam Emotion Recognition", "About"]
    activiteis = ["Αρχική","Ανάλυση συναισθήματος σε εικόνα", "Ανίχνεση προσώπου μέσω κάμερας",  "About"]

    choice = st.sidebar.selectbox("Επιλογή Ενέργειας", activiteis)
    st.sidebar.markdown(""" """)
    # if choice == "Home":
    if choice == "Αρχική":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Εφαρμογή Αναγνώρισης Συναισθημάτων Προσώπου μέσω νευρωνικών δικτών - CNN</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 Η εφαρμογή έχει 2 λειτουργίες:
                 1. Ανάλυση συναισθήματος σε μια φωτογραφία.
                 2. Αναγνώριση συναισθήματος σε πραγματικό χρόνο μέσω κάμερας.
                 """)

    # elif choice == "Webcam Emotion Recognition":
    elif choice == "Ανίχνεση προσώπου μέσω κάμερας":
        # st.header("Webcam Real-time Emotion Recognition")
        st.header("Αναγνώριση συναισθήματος σε πραγματικό χρόνο μέσω κάμερας")
        html_temp_Webcam1 = """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Εφαρμογή αναγνώρισης συναισθήματος προσώπου</h4>
                                <br>
                                <strong>Βήματα για την λειτουργία:</strong>
                                <ol type = "1">
                                <li>Επιλογή συσκευή κάμερας</li>
                                <li>Εκίνηση της κάμερας</li>
                                </ol></div></br>"""
        st.markdown(html_temp_Webcam1, unsafe_allow_html=True)
        # state = True
        stream = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                                 media_stream_constraints={"video": True, "audio": False},
                                 video_processor_factory=FaceEmotion,
                                 async_processing=True)


    # elif choice == "Analyze Image Emotion":
    elif choice == "Ανάλυση συναισθήματος σε εικόνα":
        # st.header("Analyze Image Emotion")
        # st.write("Click to upload your image ")
        st.header("Ανάλυση εικόνας")
        st.write("Κάνε κλικ στο κουτάκι εάν θες να ανεβάσεις μια εικόνα ")

        # Select a file
        # if st.checkbox('Select a file in current directory'):
        if st.checkbox('Επιλογή αρχείου από φάκελο'):
            folder_path = '.'

            # if st.checkbox('Choose directory'):
            if st.checkbox('Επιλογή φακέλου'):
                # my tests
                # folder_path = st.text_input('Enter folder path', 'D:\\EmotionRecognition\\images\\test')
                # folder_path = st.text_input('Ο φάκελος', 'D:\\EmotionRecognition\\images\\test')
                # dirnames = os.listdir('C:\\Users\\')

                folder_path = st.text_input('Ο φάκελος (π.χ C:\\Users\\User\\Pictures\\)', 'C:\\')
                try:
                    # dir_path= dir_selector()
                    # image_file = file_selector(folder_path=dir_path)
                    image_file = file_selector(folder_path=folder_path)

                    # st.write('You selected `%s`' % image_file)
                    st.write('Επέλεξες το αρχείο `%s`' % image_file)

                    # print(f"ewwe ewffwefew {image_file}")



                    col1, col2 = st.columns(2, gap='small')
                    with col1:
                        domi_emotion, bar_emotions = image_classification(image_file)

                        # show_image = cv2.imread(image_file)
                        # show_image = cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR)

                        print(domi_emotion)
                    with col2:
                        st.pyplot(bar_emotions)

                except:
                    print("Not found")


                # try:
                #     os.remove(image_file)
                # except:
                #     print("REWSAEFD")
                #     pass
            # if st.checkbox('Επιλογή φακέλου'):
            #     # my tests
            #     folder_path = st.text_input('Enter folder path', '.')
            #     # folder_path = st.text_input('Enter folder path', 'D:\\EmotionRecognition\\images\\test')
            #     # folder_path = st.text_input('Ο φάκελος', 'D:\\EmotionRecognition\\images\\test')
            #     # dirnames = os.listdir('C:\\Users\\')
            #
            #     # folder_path = st.text_input('Ο φάκελος (π.χ C:\\Users\\User\\Pictures\\)', 'C:\\')
            #     # try:
            #     # dir_path= dir_selector()
            #     # image_file = file_selector(folder_path=dir_path)
            #     image_file = file_selector(folder_path=folder_path)
            #
            #     # st.write('You selected `%s`' % image_file)
            #     st.write('Επέλεξες το αρχείο `%s`' % filename)
            #
            #     # print(f"ewwe ewffwefew {image_file}")
            #
            #     col1, col2 = st.columns(2, gap='small')
            #     with col1:
            #         domi_emotion, bar_emotions = image_classification(image_file)
            #
            #         # show_image = cv2.imread(image_file)
            #         # show_image = cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR)
            #
            #         print(domi_emotion)
            #     with col2:
            #         st.pyplot(bar_emotions)
            #     # except:
            #     #     pass

        # uploaded_file = st.file_uploader("Choose a file")
        # if uploaded_file is not None:
        #     # To read file as bytes:
        #     bytes_data = uploaded_file.read()
        #     # bytes_data = uploaded_file.getvalue()
        #     # st.write(bytes_data)
        #     decoded = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)
        #     decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        #     filename = uploaded_file.name
        #     st.write("filename:", uploaded_file.name)
        #     # cv2.imwrite()
        #     # st.image(decoded)
        #     # plt.imshow(decoded)
        #     # st.pyplot()
        #     if filename is not None:
        #         if st.checkbox('Επιλογή φακέλου'):
        #             # my tests
        #             folder_path = st.text_input('Enter folder path', '.')
        #             # folder_path = st.text_input('Enter folder path', 'D:\\EmotionRecognition\\images\\test')
        #             # folder_path = st.text_input('Ο φάκελος', 'D:\\EmotionRecognition\\images\\test')
        #             # dirnames = os.listdir('C:\\Users\\')
        #
        #             # folder_path = st.text_input('Ο φάκελος (π.χ C:\\Users\\User\\Pictures\\)', 'C:\\')
        #         # try:
        #             # dir_path= dir_selector()
        #             # image_file = file_selector(folder_path=dir_path)
        #             image_file = file_selector(folder_path=folder_path)
        #
        #             # st.write('You selected `%s`' % image_file)
        #             st.write('Επέλεξες το αρχείο `%s`' % filename)
        #
        #             # print(f"ewwe ewffwefew {image_file}")
        #
        #
        #
        #             col1, col2 = st.columns(2, gap='small')
        #             with col1:
        #                 domi_emotion, bar_emotions = image_classification(image_file)
        #
        #                 # show_image = cv2.imread(image_file)
        #                 # show_image = cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR)
        #
        #                 print(domi_emotion)
        #             with col2:
        #                 st.pyplot(bar_emotions)
        #             # except:
        #             #     pass


            # st.write(bytes_data)

        # upload = st.file_uploader("Image upload")
        # if upload:
        #     upload.getvalue()
        # # data = uploaded_file.read()
        # classify_and_localize = st.button("Classify and Localize image")
        # if classify_and_localize:
        #     st.write("")
        #     st.write("Classifying and Localizing...")

        # Multiple images
        # uploaded_file = st.file_uploader(label="Upload Files", type=['png', 'jpeg','jpg'],accept_multiple_files = True)
        #
        # # Single Image
        # uploaded_file = st.file_uploader(label="Upload Files", type=['png', 'jpeg', 'jpg'])
        # print(type(uploaded_file))
        # if uploaded_file is not None:
        #     bytes_data = uploaded_file.read()
        #     # image = Image.open(bytes_data)
        #     # image = np.fromstring(bytes_data, dtype="uint8")
        #
        #     image = np.asarray(bytearray(bytes_data))
        #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #     domi_emotion = image_classification(image)
        #     st.write(domi_emotion)
        #
        #     # bytes_data = uploaded_file[0].read()
        #     # for image in uploaded_file:
        #     #     bytes_data = image.read()
        #     inputShape = (48, 48)

            # image = cv2.imread(image,cv2. BYTES2b) .open(BytesIO(bytes_data))

    elif choice == "About":
        st.subheader("Λίγα λόγια για την εφαρμογή")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Εφαρμογή αναγνώρισης συναισθήματος στη περιοχή του προσώπου.</h4>
                                    <br>
                                    <strong>Χρησιμοποιήθηκαν:</strong>
                                    <ol type = "1">
                                    <li>Το εκπαιδευμένο μοντέλο νευρωνικού δικτύου CNN για την αναγνώριση συναισθήματος.</li>
                                    <li>Η OpenCV βιβλιοθήκη για την λειτουργία της κάμερας σε πραγματικό χρόνο.</li>
                                    <li>Το framework του streamlit για την δημιουργία της Web εφαρμογής.</li>
                                    </ol> 
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)


        html_temp4 = """
                    <div style="background-color:#98AFC7;padding:10px">
                    <h4 style="color:white;text-align:center;">Η εφαρμογή δημιουργήθηκε από τον Στέλιο Γεωργαρά</h4>
                    <h5 style="color:white;text-align:center;">Ευχαριστώ </h5>
                    </div>
                    <br></br>
                    <br></br>"""

        # st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()