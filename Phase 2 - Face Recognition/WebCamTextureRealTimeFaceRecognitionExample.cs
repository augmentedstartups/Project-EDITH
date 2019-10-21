// Phase 2 - Project E.D.I.T.H. Face Recognition
// Check the full Project E.D.I.T.H. Course Here http://bit.ly/UltimateAI50Webinar
// Modified by Ritesh Kanjee | Augmented Startups 
// Date: 18 October 2019
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.FaceModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using Rect = OpenCVForUnity.CoreModule.Rect;

namespace RealTimeFaceRecognitionExample
{
    /// <summary>
    /// WebcamTexture Real Time Face Recognition Example
    /// Detect and Recognize face in a webcam image using Eigenfaces / Fisherfaces Algorithm.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter8_FaceRecognition using "OpenCV for Unity".
    /// </summary>
    [RequireComponent (typeof(WebCamTextureToMatHelper))]
    public class WebCamTextureRealTimeFaceRecognitionExample : MonoBehaviour
    {

        public GameObject[] GUI_ID;  // GUI ID Array
        public GameObject[] sphere;  // Used for reference between CV space and Unity World Space
        
        int c_identity = -1;        //Temp variable to cap the identity to index values.
        //public GameObject sphere_ref;
        public float speed = 1f;
        private Vector3 destination;
        /// <summary>
        /// The texture.
        /// </summary>
        Texture2D texture;

        /// <summary>
        /// The web cam texture to mat helper.
        /// </summary>
        WebCamTextureToMatHelper webCamTextureToMatHelper;

        /// <summary>
        /// The face cascade.
        /// </summary>
        CascadeClassifier faceCascade;

        /// <summary>
        /// The eye cascade1.
        /// </summary>
        CascadeClassifier eyeCascade1;

        /// <summary>
        /// The eye cascade2.
        /// </summary>
        CascadeClassifier eyeCascade2;

        /// <summary>
        /// The waiting flag of the web camera initialization.
        /// </summary>
        bool isWaitingWebCameraInit = false;

        /// <summary>
        /// The coroutine to wait for the initialization of the web camera.
        /// </summary>
        Coroutine coroutineToWaitWebCameraInit;

        /// <summary>
        /// The old processing time.
        /// </summary>
        float old_processingTime = 0;

        /// <summary>
        /// The change in seconds for processing.
        /// </summary>
        const float CHANGE_IN_SECONDS_FOR_PROCESSING = 0.3f;

        /// <summary>
        /// The previous identity result.
        /// </summary>
        int prev_identity;

        /// <summary>
        /// The previous similarity result.
        /// </summary>
        double prev_similarity;

        /// <summary>
        /// The previously prepreprocessed face result.
        /// </summary>
        Mat prev_prepreprocessedFace;

        /// <summary>
        /// The reconstructed face result.
        /// </summary>
        Mat reconstructedFace;

        /// <summary>
        /// The string builder.
        /// </summary>
        StringBuilder strBuilder = new StringBuilder (100, 100);
        Scalar BLACK = new Scalar (0, 0, 0, 255);
        Scalar WHITE = new Scalar (255, 255, 255, 255);
        Scalar GREEN = new Scalar (0, 255, 0, 255);
        Scalar LIGHT_BLUE = new Scalar (0, 255, 255, 255);
        Scalar RED = new Scalar (0, 0, 255, 255);
        Scalar YELLOW = new Scalar (255, 255, 0, 255);
        Scalar LIGHT_GRAY = new Scalar (200, 200, 200, 255);
        Scalar DARK_GRAY = new Scalar (90, 90, 90, 255);


        // The Face Recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
        //    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
        //    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
        //    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006). // ...Not implemented in this example.//
        string facerecAlgorithm = "FaceRecognizer.Fisherfaces";
        //string facerecAlgorithm = "FaceRecognizer.Eigenfaces";

        const string saveDirectoryName = "RealTimeFaceRecognitionExample";

        public enum facerecAlgorithmEnumType
        {
            Fisherfaces,
            Eigenfaces,
        }

        /// <summary>
        /// The type of the facerec algorithm.
        /// </summary>
        public facerecAlgorithmEnumType facerecAlgorithmType;


        // Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
        // A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
        // conditions, and if you use a different Face Recognition algorithm.
        // Note that a higher threshold value means accepting more faces as known people,
        // whereas lower values mean more faces will be classified as "unknown".
        public float UNKNOWN_PERSON_THRESHOLD = 0.5f;


        // Cascade Classifier file, used for Face Detection.
        const string faceCascadeFilename = "lbpcascade_frontalface.xml";
        // LBP face detector.
        //const string faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
        //const string eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
        //const string eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
        //const string eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
        //const string eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
        const string eyeCascadeFilename1 = "haarcascade_eye.xml";
        // Basic eye detector for open eyes only.
        const string eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml";
        // Basic eye detector for open eyes if they might wear glasses.

        // Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
        const int faceWidth = 70;
        const int faceHeight = faceWidth;

        // Try to set the camera resolution. Note that this only works for some cameras on
        // some computers and only for some drivers, so don't rely on it to work!
        //const int DESIRED_CAMERA_WIDTH = 640;
        //const int DESIRED_CAMERA_HEIGHT = 480;

        // Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
        const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3d;
        // How much the facial image should change before collecting a new face photo for training.
        const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0d;
        // How much time must pass before collecting a new face photo for training.

        const string windowName = "WebcamFaceRec";
        // Name shown in the GUI window.
        const int BORDER = 8;
        // Border between GUI elements to the edge of the image.

        const bool preprocessLeftAndRightSeparately = true;
        // Preprocess left & right sides of the face separately, in case there is stronger light on one side.

        // Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
        bool m_debug = true;


        // Running mode for the Webcam-based interactive GUI program.
        string[] MODE_NAMES = new string [7] {
            "Startup",
            "Detection",
            "Collect Faces",
            "Training",
            "Recognition",
            "Delete All",
            "ERROR!"
        };
        MODES m_mode = MODES.MODE_STARTUP;
        int m_selectedPerson = -1;
        int m_numPersons = 0;
        List<int> m_latestFaces = new List<int> ();

        // Position of GUI buttons:
        int m_gui_faces_left = -1;
        int m_gui_faces_top = -1;

        //In recognizeAndTrainUsingWebcam function.
        BasicFaceRecognizer model;
        List<Mat> preprocessedFaces = new List<Mat> ();
        List<int> faceLabels = new List<int> ();
        Mat old_prepreprocessedFace;
        double old_time = 0.0d;

        string faceCascadeFilePath;
        string eyeCascadeFilePath1;
        string eyeCascadeFilePath2;

        #if UNITY_WEBGL && !UNITY_EDITOR
        IEnumerator getFilePath_Coroutine;
        #endif

        // Use this for initialization
        void Start ()
        {
            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper> ();

            for (int i =0; i < GUI_ID.Length; i++) // Initally set all GUI Identities to Inactive.
                { 
                    GUI_ID[i].SetActive(false);
                }

            #if UNITY_WEBGL && !UNITY_EDITOR
            getFilePath_Coroutine = GetFilePath ();
            StartCoroutine (getFilePath_Coroutine);
            #else
            faceCascadeFilePath = Utils.getFilePath (faceCascadeFilename);
            eyeCascadeFilePath1 = Utils.getFilePath (eyeCascadeFilename1);
            eyeCascadeFilePath2 = Utils.getFilePath (eyeCascadeFilename2);

            Run ();
            #endif
        }

        #if UNITY_WEBGL && !UNITY_EDITOR
        // wait to gets file paths.
        private IEnumerator GetFilePath()
        {
            var getFilePathAsync_faceCascade_Coroutine = Utils.getFilePathAsync (faceCascadeFilename, (result) => {
                faceCascadeFilePath = result;
            });
            yield return getFilePathAsync_faceCascade_Coroutine;

            var getFilePathAsync_eyeCascade1_Coroutine = Utils.getFilePathAsync (eyeCascadeFilename1, (result) => {
                eyeCascadeFilePath1 = result;
            });
            yield return getFilePathAsync_eyeCascade1_Coroutine;

            var getFilePathAsync_eyeCascade2_Coroutine = Utils.getFilePathAsync (eyeCascadeFilename2, (result) => {
                eyeCascadeFilePath2 = result;
            });
            yield return getFilePathAsync_eyeCascade2_Coroutine;

            getFilePath_Coroutine = null;

            Run();
        }
        #endif

        private void Run ()
        {
            if (facerecAlgorithmType == facerecAlgorithmEnumType.Fisherfaces) {
                facerecAlgorithm = "FaceRecognizer.Fisherfaces";
            } else if (facerecAlgorithmType == facerecAlgorithmEnumType.Eigenfaces) {
                facerecAlgorithm = "FaceRecognizer.Eigenfaces";
            }

            // Load the face and 1 or 2 eye detection XML classifiers.
            initDetectors (ref faceCascade, ref eyeCascade1, ref eyeCascade2);

            // Since we have already initialized everything, lets start in Detection mode.
            m_mode = MODES.MODE_DETECTION;

            coroutineToWaitWebCameraInit = StartCoroutine (waitWebCameraInit (false, 0, 0.5f));
        }

        /// <summary>
        /// Raises the web cam texture to mat helper inited event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInited ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperInited");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat ();

            if (texture == null || (texture.width != webCamTextureMat.cols () || texture.height != webCamTextureMat.rows ()))
                texture = new Texture2D (webCamTextureMat.cols (), webCamTextureMat.rows (), TextureFormat.RGBA32, false);

            gameObject.transform.localScale = new Vector3 (webCamTextureMat.cols (), webCamTextureMat.rows (), 1);
            Debug.Log ("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            float width = gameObject.transform.localScale.x;
            float height = gameObject.transform.localScale.y;

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale) {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            } else {
                Camera.main.orthographicSize = height / 2;
            }

            gameObject.GetComponent<Renderer> ().material.mainTexture = texture;
        }

        /// <summary>
        /// Raises the web cam texture to mat helper disposed event.
        /// </summary>
        public void OnWebCamTextureToMatHelperDisposed ()
        {
            Debug.Log ("OnWebCamTextureToMatHelperDisposed");
        }

        /// <summary>
        /// Raises the web cam texture to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        public void OnWebCamTextureToMatHelperErrorOccurred (WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log ("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }
            
        // Update is called once per frame
        void Update ()
        {
            if (webCamTextureToMatHelper.IsPlaying () && webCamTextureToMatHelper.DidUpdateThisFrame ()) {

                Mat rgbaMat = webCamTextureToMatHelper.GetMat ();
                Color32[] colors = webCamTextureToMatHelper.GetBufferColors ();

                if (rgbaMat != null) {
                    if (!isWaitingWebCameraInit) {
                        // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
                        recognizeAndTrainUsingWebcam (rgbaMat, faceCascade, eyeCascade1, eyeCascade2);
                    }

                    Utils.matToTexture2D (rgbaMat, texture, colors);
                }
            }
                       
            switch (c_identity) // Smoothly transform the position and scale of the Identifier to the pose of the recognized face. 
            {
                case 2:
                    GUI_ID[2].transform.position = Vector3.MoveTowards(GUI_ID[2].transform.position, sphere[2].transform.position, Time.deltaTime * speed);
                    GUI_ID[2].transform.localScale = Vector3.Lerp(GUI_ID[2].transform.localScale, sphere[2].transform.localScale, Time.deltaTime * speed); 
                    break;
                case 1:
                    GUI_ID[1].transform.position = Vector3.MoveTowards(GUI_ID[1].transform.position, sphere[1].transform.position, Time.deltaTime * speed);
                    GUI_ID[1].transform.localScale = Vector3.Lerp(GUI_ID[2].transform.localScale, sphere[1].transform.localScale, Time.deltaTime * speed);
                    break;
                case 0:
                    GUI_ID[0].transform.position = Vector3.MoveTowards(GUI_ID[0].transform.position, sphere[0].transform.position, Time.deltaTime * speed);
                    GUI_ID[0].transform.localScale = Vector3.Lerp(GUI_ID[0].transform.localScale, sphere[0].transform.localScale, Time.deltaTime * speed);
                    break;
                default:
                    break;
            }




            if (Input.GetMouseButtonUp (0)) {
                // Check if the user clicked on one of our GUI buttons..
                if (isUGUIHit (Input.mousePosition))
                    return;

                RaycastHit hit;
                if (!Physics.Raycast (Camera.main.ScreenPointToRay (Input.mousePosition), out hit))
                    return;

                Vector2 pixelUV = hit.textureCoord;

                _onMouseUP ((int)(texture.width * pixelUV.x), (int)(texture.height * (1 - pixelUV.y)));
            }
        }

        // wait web camera initialization.
        private IEnumerator waitWebCameraInit (bool isChangeCamera = false, float beforWaitTime = 0, float afterWaitTime = 0)
        {
            if (isWaitingWebCameraInit)
                yield break;

            isWaitingWebCameraInit = true;
            if (beforWaitTime > 0)
                yield return new WaitForSeconds (beforWaitTime);

            if (isChangeCamera)
                webCamTextureToMatHelper.Initialize (null, webCamTextureToMatHelper.requestedWidth, webCamTextureToMatHelper.requestedHeight, !webCamTextureToMatHelper.requestedIsFrontFacing);
            else
                webCamTextureToMatHelper.Initialize ();

            if (afterWaitTime > 0)
                yield return new WaitForSeconds (afterWaitTime);

            isWaitingWebCameraInit = false;
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        void OnDestroy ()
        {
            webCamTextureToMatHelper.Dispose ();

            if (faceCascade != null && !faceCascade.IsDisposed)
                faceCascade.Dispose ();

            if (eyeCascade1 != null && !eyeCascade1.IsDisposed)
                eyeCascade1.Dispose ();

            if (eyeCascade2 != null && !eyeCascade2.IsDisposed)
                eyeCascade2.Dispose ();

            foreach (Mat face in preprocessedFaces) {
                if (face != null && !face.IsDisposed)
                    face.Dispose ();
            }

            if (old_prepreprocessedFace != null && !old_prepreprocessedFace.IsDisposed)
                old_prepreprocessedFace.Dispose ();

            if (prev_prepreprocessedFace != null && !prev_prepreprocessedFace.IsDisposed)
                prev_prepreprocessedFace.Dispose ();

            if (reconstructedFace != null && !reconstructedFace.IsDisposed)
                reconstructedFace.Dispose ();

            if (model != null && !model.IsDisposed)
                model.Dispose ();

            
            #if UNITY_WEBGL && !UNITY_EDITOR
            if (getFilePath_Coroutine != null) {
                StopCoroutine (getFilePath_Coroutine);
                ((IDisposable)getFilePath_Coroutine).Dispose ();
            }
            #endif
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick ()
        {
            SceneManager.LoadScene ("RealTimeFaceRecognitionExample");
        }

        /// <summary>
        /// Raises the play button click event.
        /// </summary>
        public void OnPlayButtonClick ()
        {
            webCamTextureToMatHelper.Play ();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public void OnPauseButtonClick ()
        {
            if (isWaitingWebCameraInit)
                StopCoroutine (coroutineToWaitWebCameraInit);
            
            webCamTextureToMatHelper.Pause ();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public void OnStopButtonClick ()
        {
            if (isWaitingWebCameraInit)
                StopCoroutine (coroutineToWaitWebCameraInit);
            
            webCamTextureToMatHelper.Stop ();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public void OnChangeCameraButtonClick ()
        {
            if (isWaitingWebCameraInit)
                return;

            coroutineToWaitWebCameraInit = StartCoroutine (waitWebCameraInit (true, 0.5f, 0.5f));
        }

        /// <summary>
        /// Raises the add person button click event.
        /// </summary>
        public void OnAddPersonButtonClick ()
        {
            Debug.Log ("User clicked [Add Person] button when numPersons was " + m_numPersons + ".");

            // Check if there is already a person without any collected faces, then use that person instead.
            // This can be checked by seeing if an image exists in their "latest collected face".
            if ((m_numPersons == 0) || (m_latestFaces [m_numPersons - 1] >= 0)) {
                // Add a new person.
                m_numPersons++;
                m_latestFaces.Add (-1); // Allocate space for an extra person.
                Debug.Log ("Num Persons: " + m_numPersons + ".");
            }
            // Use the newly added person. Also use the newest person even if that person was empty.
            m_selectedPerson = m_numPersons - 1;
            m_mode = MODES.MODE_COLLECT_FACES;
        }

        /// <summary>
        /// Raises the delete all button click event.
        /// </summary>
        public void OnDeleteAllButtonClick ()
        {
            Debug.Log ("User clicked [Delete All] button.");
            m_mode = MODES.MODE_DELETE_ALL;
        }

        /*
        /// <summary>
        /// Raises the show debug toggle value changed event.
        /// </summary>
        public void OnShowDebugToggleValueChanged()
        {
            Debug.Log("User clicked [Debug] button.");
            m_debug = !m_debug;
            Debug.Log("Debug mode: " + m_debug + ".");
        }
        */

        /// <summary>
        /// Raises the save button click event.
        /// </summary>
        public void OnSaveButtonClick ()
        {
            Debug.Log ("User clicked [Save] button.");

            string saveDirectoryPath = Path.Combine (Application.persistentDataPath, saveDirectoryName);

            if (model != null) {
                // Clean up old files.
                if (Directory.Exists (saveDirectoryPath)) {
                    DirectoryInfo directoryInfo = new DirectoryInfo (saveDirectoryPath);
                    foreach (FileInfo fileInfo in directoryInfo.GetFiles()) {
                        if ((fileInfo.Attributes & FileAttributes.ReadOnly) == FileAttributes.ReadOnly) {
                            fileInfo.Attributes = FileAttributes.Normal;
                        }
                    }
                    if ((directoryInfo.Attributes & FileAttributes.ReadOnly) == FileAttributes.ReadOnly) {
                        directoryInfo.Attributes = FileAttributes.Directory;
                    }
                    directoryInfo.Delete (true);
                }
                Directory.CreateDirectory (saveDirectoryPath);

                // save the train data.
                model.write (Path.Combine (saveDirectoryPath, "traindata.yml"));

                // save the preprocessedfaces.
                #if UNITY_WEBGL && !UNITY_EDITOR
                string format = "jpg";
                MatOfInt compressionParams = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 100);
                #else
                string format = "png";
                MatOfInt compressionParams = new MatOfInt (Imgcodecs.IMWRITE_PNG_COMPRESSION, 0);
                #endif
                for (int i = 0; i < m_numPersons; ++i) {
                    Imgcodecs.imwrite (Path.Combine (saveDirectoryPath, "preprocessedface" + i + "." + format), preprocessedFaces [m_latestFaces [i]], compressionParams);
                }
            } else {
                Debug.Log ("save failure. train data does not exist.");
            }
        }

        /// <summary>
        /// Raises the load button click event.
        /// </summary>
        public void OnLoadButtonClick ()
        {
            Debug.Log ("User clicked [Load] button.");

            string loadDirectoryPath = Path.Combine (Application.persistentDataPath, saveDirectoryName);

            if (!Directory.Exists (loadDirectoryPath)) {
                Debug.Log ("load failure. saved train data file does not exist.");
                return;
            }

            // Restart everything!
            dispose ();

            if (facerecAlgorithm == "FaceRecognizer.Fisherfaces") {
                model = FisherFaceRecognizer.create ();
            } else if (facerecAlgorithm == "FaceRecognizer.Eigenfaces") {
                model = EigenFaceRecognizer.create ();
            }

            if (model == null) {
                Debug.LogError ("ERROR: The FaceRecognizer algorithm [" + facerecAlgorithm + "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer.");
                m_mode = MODES.MODE_DETECTION;
                return;
            }

            // load the train data.
            model.read (Path.Combine (loadDirectoryPath, "traindata.yml"));

            int maxLabel = (int)Core.minMaxLoc (model.getLabels ()).maxVal;

            if (maxLabel < 0) {
                Debug.Log ("load failure.");
                model.Dispose ();
                model = null;
                m_mode = MODES.MODE_DETECTION;
                return;
            }

            // Restore the save data.
            #if UNITY_WEBGL && !UNITY_EDITOR
            string format = "jpg";
            #else
            string format = "png";
            #endif
            m_numPersons = maxLabel + 1;
            for (int i = 0; i < m_numPersons; ++i) {
                m_latestFaces.Add (i);
                preprocessedFaces.Add (Imgcodecs.imread (Path.Combine (loadDirectoryPath, "preprocessedface" + i + "." + format), 0));
                if (preprocessedFaces [i].total () == 0)
                    preprocessedFaces [i] = new Mat (faceHeight, faceWidth, CvType.CV_8UC1, new Scalar (128));
                faceLabels.Add (i);
            }

            // go to the recognition mode!
            m_mode = MODES.MODE_RECOGNITION;
        }

        // Load the face and 1 or 2 eye detection XML classifiers.
        private void initDetectors (ref CascadeClassifier faceCascade, ref CascadeClassifier eyeCascade1, ref CascadeClassifier eyeCascade2)
        {
            faceCascade = new CascadeClassifier (faceCascadeFilePath);
//            if (faceCascade.empty ()) {
//                Debug.LogError ("cascade file is not loaded.Please copy from “RealTimeFaceRecognitionExample/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
//            }

            eyeCascade1 = new CascadeClassifier (eyeCascadeFilePath1);
//            if (eyeCascade1.empty ()) {
//                Debug.LogError ("cascade file is not loaded.Please copy from “RealTimeFaceRecognitionExample/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
//            }

            eyeCascade2 = new CascadeClassifier (eyeCascadeFilePath2);
//            if (eyeCascade2.empty ()) {
//                Debug.LogError ("cascade file is not loaded.Please copy from “RealTimeFaceRecognitionExample/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
//            }
        }

        // Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
        // and/or negative y coords for bottom-justified text.
        // Returns the bounding rect around the drawn text.
        private Rect drawString (Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = Imgproc.FONT_HERSHEY_TRIPLEX)
        {
            // Get the text size & baseline.
            int[] baseline = new int[1] { 0 };
            Size textSize = Imgproc.getTextSize (text, fontFace, fontScale, thickness, baseline);
            baseline [0] += thickness;

            // Adjust the coords for left/right-justified or top/bottom-justified.
            if (coord.y >= 0) {
                // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
                coord.y += textSize.height;
            } else {
                // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
                coord.y += img.rows () - baseline [0] + 1;
            }
            // Become right-justified if desired.
            if (coord.x < 0) {
                coord.x += img.cols () - textSize.width + 1;
            }

            // Get the bounding box around the text.
            Rect boundingRect = new Rect ((int)coord.x, (int)(coord.y - textSize.height), (int)textSize.width, (int)(baseline [0] + textSize.height));

            // Draw anti-aliased text.
            //Imgproc.putText (img, text, coord, fontFace, fontScale, color, thickness, Imgproc.LINE_AA, false);

            // Let the user know how big their text is, in case they want to arrange things.
            return boundingRect;
        }

        // determines whether the point is included in the rectangle.
        private bool isPointInRect (Point pt, Rect rc)
        {
            if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
            if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
                return true;

            return false;
        }

        // Mouse event handler. Called automatically by OpenCV when the user clicks in the GUI window.
        private void _onMouseUP (int x, int y)
        {
            Debug.Log ("_onMouseUP() x:" + x + " y:" + y);

            Point pt = new Point (x, y);
            Debug.Log ("User clicked on the image");
            // Check if the user clicked on one of the faces in the list.
            int clickedPerson = -1;
            for (int i = 0; i < m_numPersons; i++) {
                if (m_gui_faces_top >= 0) {
                    Rect rcFace = new Rect (m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                    if (isPointInRect (pt, rcFace)) {
                        clickedPerson = i;
                        break;
                    }
                }
            }
            // Change the selected person, if the user clicked on a face in the GUI.
            if (clickedPerson >= 0) {
                // Change the current person, and collect more photos for them.
                m_selectedPerson = clickedPerson; // Use the newly added person.
                m_mode = MODES.MODE_COLLECT_FACES;
            }
                // Otherwise they clicked in the center.
            else {
                // Change to training mode if it was collecting faces.
                if (m_mode == MODES.MODE_COLLECT_FACES) {
                    Debug.Log ("User wants to begin training.");
                    m_mode = MODES.MODE_TRAINING;
                }
            }

            //Debug.Log("m_mode:" + m_mode + " clickedPerson:" + clickedPerson);
        }

        // determines whether or not the mouse position is hitting the UGUI.
        private bool isUGUIHit (Vector3 _scrPos)
        {
            // Input.mousePosition
            PointerEventData pointer = new PointerEventData (EventSystem.current);
            pointer.position = _scrPos;
            List<RaycastResult> result = new List<RaycastResult> ();
            EventSystem.current.RaycastAll (pointer, result);
            return (result.Count > 0);
        }
            
        // Main loop that runs forever, until the user hits Escape to quit.
        private void recognizeAndTrainUsingWebcam (Mat cameraFrame, CascadeClassifier faceCascade, CascadeClassifier eyeCascade1, CascadeClassifier eyeCascade2)
        {
            if (cameraFrame != null && cameraFrame.total () == 0) {
                Debug.LogError ("ERROR: Couldn't grab the next camera frame.");
            }

            // Get a copy of the camera frame that we can draw onto.
            Mat displayedFrame = cameraFrame;

            int cx;
            float current_processingTime = Time.realtimeSinceStartup;
            float processingTimeDiff_seconds = (current_processingTime - old_processingTime);
            if (processingTimeDiff_seconds > CHANGE_IN_SECONDS_FOR_PROCESSING) {

                // Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
                int identity = -1;

                // Find a face and preprocess it to have a standard size and contrast & brightness.
                Rect faceRect = new Rect ();  // Position of detected face.
                Rect searchedLeftEye = new Rect (), searchedRightEye = new Rect (); // top-left and top-right regions of the face, where eyes were searched.
                Point leftEye = new Point (), rightEye = new Point ();    // Position of the detected eyes.

                Mat preprocessedFace = PreprocessFace.GetPreprocessedFace (displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, ref faceRect, ref leftEye, ref rightEye, ref searchedLeftEye, ref searchedRightEye);

                bool gotFaceAndEyes = false;

                if (preprocessedFace != null && preprocessedFace.total () > 0)
                    gotFaceAndEyes = true;

                // Draw an anti-aliased rectangle around the detected face.
                if (faceRect.width > 0) {
                    Imgproc.rectangle (displayedFrame, faceRect.tl (), faceRect.br (), LIGHT_BLUE, 2, Imgproc.LINE_AA, 0);
                    
                    // Draw light-blue anti-aliased circles for the 2 eyes.
                    Scalar eyeColor = LIGHT_BLUE;
                    if (leftEye.x >= 0) {   // Check if the eye was detected
                        Imgproc.circle (displayedFrame, new Point (faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, Imgproc.LINE_AA, 0);
                    }
                    if (rightEye.x >= 0) {   // Check if the eye was detected
                        Imgproc.circle (displayedFrame, new Point (faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, Imgproc.LINE_AA, 0);
                    }

               
                    
                    
                }

                prev_prepreprocessedFace = preprocessedFace;

                if (m_mode == MODES.MODE_DETECTION) {
                    // Don't do anything special.
                } else if (m_mode == MODES.MODE_COLLECT_FACES) {

                    // Check if we have detected a face.
                    if (gotFaceAndEyes) {
                        // Check if this face looks somewhat different from the previously collected face.
                        double imageDiff = 10000000000.0d;
                        if (old_prepreprocessedFace != null && old_prepreprocessedFace.total () > 0) {
                            imageDiff = Recognition.GetSimilarity (preprocessedFace, old_prepreprocessedFace);
                        }

                        // Also record when it happened.
                        double current_time = Time.realtimeSinceStartup;
                        double timeDiff_seconds = (current_time - old_time);

                        // Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
                        if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
                            // Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
                            Mat mirroredFace = new Mat ();
                            Core.flip (preprocessedFace, mirroredFace, 1);

                            // Add the face images to the list of detected faces.
                            preprocessedFaces.Add (preprocessedFace);
                            preprocessedFaces.Add (mirroredFace);
                            faceLabels.Add (m_selectedPerson);
                            faceLabels.Add (m_selectedPerson);

                            // Keep a reference to the latest face of each person.
                            m_latestFaces [m_selectedPerson] = preprocessedFaces.Count - 2;  // Point to the non-mirrored face.
                            // Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
                            Debug.Log ("Saved face " + (preprocessedFaces.Count / 2) + " for person " + m_selectedPerson);

                            // Make a white flash on the face, so the user knows a photo has been taken.
                            using (Mat displayedFaceRegion = new Mat (displayedFrame, faceRect)) {
                                Core.add (displayedFaceRegion, DARK_GRAY, displayedFaceRegion);
                            }

                            // Keep a copy of the processed face, to compare on next iteration.
                            old_prepreprocessedFace = preprocessedFace;
                            old_time = current_time;
                        }
                    }

                } else if (m_mode == MODES.MODE_TRAINING) {
                    // Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for Fisherfaces,
                    // we need atleast 2 people otherwise it will crash!
                    bool haveEnoughData = true;
                    if (facerecAlgorithm == "FaceRecognizer.Fisherfaces") {
                        if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces [1] < 0)) {
                            Debug.Log ("Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ...");
                            haveEnoughData = false;
                        }
                    }
                    if (m_numPersons < 1 || preprocessedFaces.Count <= 0 || preprocessedFaces.Count != faceLabels.Count) {
                        Debug.Log ("Warning: Need some training data before it can be learnt! Collect more data ...");
                        haveEnoughData = false;
                    }

                    if (haveEnoughData) {
                        // Start training from the collected faces using Eigenfaces or a similar algorithm.
                        model = Recognition.LearnCollectedFaces (preprocessedFaces, faceLabels, facerecAlgorithm);

                        // Show the internal face recognition data, to help debugging.
                        //if (m_debug)
                        //Recognition.ShowTrainingDebugData(model, faceWidth, faceHeight);

                        // Now that training is over, we can start recognizing!
                        m_mode = MODES.MODE_RECOGNITION;
                    } else {
                        // Since there isn't enough training data, go back to the face collection mode!
                        m_mode = MODES.MODE_COLLECT_FACES;
                    }

                } else if (m_mode == MODES.MODE_RECOGNITION) {
                    prev_identity = -1;
                    prev_similarity = 100000000.0d;
                    if (reconstructedFace != null && !reconstructedFace.IsDisposed) {
                        reconstructedFace.Dispose ();
                    }
                    reconstructedFace = null;

                    if (gotFaceAndEyes && (preprocessedFaces.Count > 0) && (preprocessedFaces.Count == faceLabels.Count)) {

                        // Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
                        reconstructedFace = Recognition.ReconstructFace (model, preprocessedFace);

                        // Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
                        double similarity = Recognition.GetSimilarity (preprocessedFace, reconstructedFace);
                        double confidence = 0.0d;

                        string outputStr;
                        if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                            int[] predictedLabel = new int [1];
                            double[] predictedConfidence = new double [1];
                            // Identify who the person is in the preprocessed face image.
                            model.predict (preprocessedFace, predictedLabel, predictedConfidence);
                            identity = predictedLabel [0];

                            if (identity >= 0 && identity < GUI_ID.Length)
                            {
                                c_identity = identity;
                            }
                            else { c_identity = -1; }
                                                        
                            if (c_identity != -1)
                            {
                                GUI_ID[c_identity].SetActive(true); // Set GUI ID active when detected
                            }
                            confidence = predictedConfidence[0];
                            //GUI_ID[0].transform.localScale = new Vector3(Convert.ToSingle(100.0f-(faceRect.y)/1000),  Convert.ToSingle(100.0f- (faceRect.y)/1000), 1);

                            outputStr = identity.ToString ();
                            prev_identity = identity;

                            switch (c_identity)  // Smoohtly move the GUI to the position of reference sphere (destination).
                            {
                                case 2:
                                    sphere[2].transform.position = new Vector3(faceRect.x - 232, -faceRect.y + 300, sphere[2].transform.position.z);  // Position of the face/eyes relative to world coordinates
                                    sphere[2].transform.localScale = new Vector3((0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f), (0.0018f * (1000.0f - (float)faceRect.x) + 0.2333f) - 0.1f, 0);  // Position of the face/eyes relative to world coordinates
                                    Debug.Log("Scale : " + (0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f));
                                    break;
                                case 1:
                                    sphere[1].transform.position = new Vector3(faceRect.x - 232, -faceRect.y + 300, sphere[1].transform.position.z);  // Position of the face/eyes relative to world coordinates
                                    sphere[1].transform.localScale = new Vector3((0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f), (0.0018f * (1000.0f - (float)faceRect.x) + 0.2333f) - 0.1f, 0);  // Position of the face/eyes relative to world coordinates
                                    Debug.Log("Scale : " + (0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f));
                                    break;
                                case 0:
                                    sphere[0].transform.position = new Vector3(faceRect.x - 232, -faceRect.y + 300, sphere[0].transform.position.z);  // Position of the face/eyes relative to world coordinates
                                    sphere[0].transform.localScale = new Vector3((0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f), (0.0018f * (1000.0f - (float)faceRect.x) + 0.2333f) - 0.1f, 0);  // Position of the face/eyes relative to world coordinates
                                    Debug.Log("Scale : " + (0.0015f * (1000.0f - (float)faceRect.x) + 0.2333f));
                                    break;
                                default:
                                    print("Unknown");
                                    break;
                            }
                        } else {
                            // Since the confidence is low, assume it is an unknown person.
                            outputStr = "Unknown";
                        }
                        prev_similarity = similarity;
                        Debug.Log ("Identity: " + outputStr + ". Similarity: " + similarity + ". Confidence: " + confidence);
                        
                    }

                } else if (m_mode == MODES.MODE_DELETE_ALL) {

                    // Restart everything!
                    dispose ();

                    // Restart in Detection mode.
                    m_mode = MODES.MODE_DETECTION;

                } else {
                    Debug.LogError ("ERROR: Invalid run mode " + m_mode);
                    //exit(1);
                }

                old_processingTime = current_processingTime;
            }

            // Show the help, while also showing the number of collected faces. Since we also collect mirrored faces, we should just
            // tell the user how many faces they think we saved (ignoring the mirrored faces), hence divide by 2.
            strBuilder.Length = 0;
            Rect rcHelp = new Rect ();
            if (m_mode == MODES.MODE_DETECTION) {
                strBuilder.Append ("Click [Add Person] when ready to collect faces.");
            } else if (m_mode == MODES.MODE_COLLECT_FACES) {
                strBuilder.Append ("Click anywhere to train from your ");
                strBuilder.Append (preprocessedFaces.Count / 2);
                strBuilder.Append (" faces of ");
                strBuilder.Append (m_numPersons);
                strBuilder.Append (" People.");
            } else if (m_mode == MODES.MODE_TRAINING) {
                strBuilder.Append ("Please wait while your ");
                strBuilder.Append (preprocessedFaces.Count / 2);
                strBuilder.Append (" faces of ");
                strBuilder.Append (m_numPersons);
                strBuilder.Append (" People Builds.");
            } else if (m_mode == MODES.MODE_RECOGNITION)
                strBuilder.Append ("Click people on the right to add more faces to them, or [Add Person] for someone new.");
            
            if (strBuilder.Length > 0) {
                // Draw it with a black background and then again with a white foreground.
                // Since BORDER may be 0 and we need a negative position, subtract 2 from the border so it is always negative.
                float txtSize = 0.4f;
                drawString (displayedFrame, strBuilder.ToString (), new Point (BORDER, -BORDER - 2), BLACK, txtSize); // Black shadow.
                rcHelp = drawString (displayedFrame, strBuilder.ToString (), new Point (BORDER + 1, -BORDER - 1), WHITE, txtSize); // White text.
            }

            // Show the current mode.
            strBuilder.Length = 0;
            if (m_mode >= 0 && m_mode < MODES.MODE_END) {
                strBuilder.Append (" People builds.");
                strBuilder.Append (MODE_NAMES [(int)m_mode]);
                drawString (displayedFrame, strBuilder.ToString (), new Point (BORDER, -BORDER - 2 - rcHelp.height), BLACK); // Black shadow
                drawString (displayedFrame, strBuilder.ToString (), new Point (BORDER + 1, -BORDER - 1 - rcHelp.height), GREEN); // Green text
            }

            // Show the current preprocessed face in the top-center of the display.
            cx = (displayedFrame.cols () - faceWidth) / 2;
            if (prev_prepreprocessedFace != null && prev_prepreprocessedFace.total () > 0) {
                // Get a RGBA version of the face, since the output is RGBA color.
                using (Mat srcRGBA = new Mat (prev_prepreprocessedFace.size (), CvType.CV_8UC4)) {
                    Imgproc.cvtColor (prev_prepreprocessedFace, srcRGBA, Imgproc.COLOR_GRAY2RGBA);
                    // Get the destination ROI (and make sure it is within the image!).
                    Rect dstRC = new Rect (cx, BORDER, faceWidth, faceHeight);
                    using (Mat dstROI = new Mat (displayedFrame, dstRC)) {
                        // Copy the pixels from src to dst.
                        srcRGBA.copyTo (dstROI);
                    }
                }
            }

            // Draw an anti-aliased border around the face, even if it is not shown.
            Imgproc.rectangle (displayedFrame, new Point (cx - 1, BORDER - 1), new Point (cx - 1 + faceWidth + 2, BORDER - 1 + faceHeight + 2), LIGHT_GRAY, 1, Imgproc.LINE_AA, 0);

            // Show the most recent face for each of the collected people, on the right side of the display.
            m_gui_faces_left = displayedFrame.cols () - BORDER - faceWidth;
            m_gui_faces_top = BORDER;
            for (int i = 0; i < m_numPersons; i++) {
                int index = m_latestFaces [i];
                if (index >= 0 && index < preprocessedFaces.Count) {
                    Mat srcGray = preprocessedFaces [index];
                    if (srcGray != null && srcGray.total () > 0) {
                        // Get a RGBA version of the face, since the output is RGBA color.
                        using (Mat srcRGBA = new Mat (srcGray.size (), CvType.CV_8UC4)) {
                            Imgproc.cvtColor (srcGray, srcRGBA, Imgproc.COLOR_GRAY2RGBA);
                            // Get the destination ROI (and make sure it is within the image!).
                            int y = Mathf.Min (m_gui_faces_top + i * faceHeight, displayedFrame.rows () - faceHeight);
                            Rect dstRC = new Rect (m_gui_faces_left, y, faceWidth, faceHeight);
                            using (Mat dstROI = new Mat (displayedFrame, dstRC)) {
                                // Copy the pixels from src to dst.
                                srcRGBA.copyTo (dstROI);
                            }
                        }
                    }
                }
            }

            // Highlight the person being collected, using a red rectangle around their face.
            if (m_mode == MODES.MODE_COLLECT_FACES) {
                if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                    int y = Mathf.Min (m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows () - faceHeight);
                    Rect rc = new Rect (m_gui_faces_left, y, faceWidth, faceHeight);
                    Imgproc.rectangle (displayedFrame, rc.tl (), rc.br (), RED, 3, Imgproc.LINE_AA, 0);
                }
            }

            // Highlight the person that has been recognized, using a green rectangle around their face.
            if (m_mode == MODES.MODE_RECOGNITION && prev_identity >= 0 && prev_identity < 1000) {
                int y = Mathf.Min (m_gui_faces_top + prev_identity * faceHeight, displayedFrame.rows () - faceHeight);
                Rect rc = new Rect (m_gui_faces_left, y, faceWidth, faceHeight);
                Imgproc.rectangle (displayedFrame, rc.tl (), rc.br (), GREEN, 3, Imgproc.LINE_AA, 0);
            }

            if (m_mode == MODES.MODE_RECOGNITION) {
                if (m_debug) {
                    if (reconstructedFace != null && reconstructedFace.total () > 0) {
                        cx = (displayedFrame.cols () - faceWidth) / 2;
                        Point rfDebugBottomRight = new Point (cx + faceWidth * 2 + 5, BORDER + faceHeight);
                        Point rfDebugTopLeft = new Point (cx + faceWidth + 5, BORDER);
                        Rect rfDebugRC = new Rect (rfDebugTopLeft, rfDebugBottomRight);
                        using (Mat srcRGBA = new Mat (reconstructedFace.size (), CvType.CV_8UC4)) {
                            Imgproc.cvtColor (reconstructedFace, srcRGBA, Imgproc.COLOR_GRAY2RGBA);
                            using (Mat dstROI = new Mat (displayedFrame, rfDebugRC)) {
                                srcRGBA.copyTo (dstROI);
                            }
                        }
                        Imgproc.rectangle (displayedFrame, rfDebugTopLeft, rfDebugBottomRight, LIGHT_GRAY, 1, Imgproc.LINE_AA, 0);
                    }
                }

                // Show the confidence rating for the recognition in the mid-top of the display.
                cx = (displayedFrame.cols () - faceWidth) / 2;
                Point ptBottomRight = new Point (cx - 5, BORDER + faceHeight);
                Point ptTopLeft = new Point (cx - 15, BORDER);
                // Draw a gray line showing the threshold for an "unknown" person.
                Point ptThreshold = new Point (ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                Imgproc.rectangle (displayedFrame, ptThreshold, new Point (ptBottomRight.x, ptThreshold.y), LIGHT_GRAY, 1, Imgproc.LINE_AA, 0);
                // Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                double confidenceRatio = 1.0d - Math.Min (Math.Max (prev_similarity, 0.0d), 1.0d);
                Point ptConfidence = new Point (ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                // Show the light-blue confidence bar.
                Imgproc.rectangle (displayedFrame, ptConfidence, ptBottomRight, LIGHT_BLUE, Core.FILLED, Imgproc.LINE_AA, 0);
                // Show the gray border of the bar.
                Imgproc.rectangle (displayedFrame, ptTopLeft, ptBottomRight, LIGHT_GRAY, 1, Imgproc.LINE_AA, 0);

            }

            /*
            // If the user wants all the debug data, show it to them!
            if (m_debug)
            {
                Mat face = new Mat();
                if (faceRect.width > 0)
                {
                    face = new Mat(cameraFrame, faceRect);
                    if (searchedLeftEye.width > 0 && searchedRightEye.width > 0)
                    {
                        Mat topLeftOfFace = new Mat(face, searchedLeftEye);
                        Mat topRightOfFace = new Mat(face, searchedRightEye);
                        //imshow("topLeftOfFace", topLeftOfFace);
                        //imshow("topRightOfFace", topRightOfFace);
                    }
                }

                //if (model != null)
                    //showTrainingDebugData(model, faceWidth, faceHeight);
            }
            */
        }

        // Disposal to restart all.
        private void dispose ()
        {
            
            m_selectedPerson = -1;
            m_numPersons = 0;
            m_latestFaces.Clear ();
            faceLabels.Clear ();
            prev_identity = -1;
            prev_similarity = 100000000.0d;

            foreach (Mat face in preprocessedFaces) {
                if (face != null && !face.IsDisposed)
                    face.Dispose ();
            }
            preprocessedFaces.Clear ();

            if (old_prepreprocessedFace != null && !old_prepreprocessedFace.IsDisposed) {
                old_prepreprocessedFace.Dispose ();
            }
            old_prepreprocessedFace = null;

            if (prev_prepreprocessedFace != null && !prev_prepreprocessedFace.IsDisposed) {
                prev_prepreprocessedFace.Dispose ();
            }
            prev_prepreprocessedFace = null;

            if (reconstructedFace != null && !reconstructedFace.IsDisposed) {
                reconstructedFace.Dispose ();
            }
            reconstructedFace = null;

            if (model != null && !model.IsDisposed) {
                model.Dispose ();
                model = null;
            }
        }
    }

    enum MODES
    {
        MODE_STARTUP,
        MODE_DETECTION,
        MODE_COLLECT_FACES,
        MODE_TRAINING,
        MODE_RECOGNITION,
        MODE_DELETE_ALL,
        MODE_END
    }
}
