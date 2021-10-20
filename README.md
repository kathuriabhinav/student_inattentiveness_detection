# ivp-online-proctoring-system

ivp-online-proctoring-system

Instructions to run this project

1. we used anaconda environments to manage dependencies for this project so to begin with download anaconda installer from [here](https://www.anaconda.com/products/individual#Downloads) .
2. create a new environment and install the following dependent python libraries in the new environment using `conda` or `pip` as `pip install "libraryname"` for the following libraries.

```
cmake
scipy
opencv
dlib
imutils
mediapipe
tensorflow==2.6.0
```

YOU COULD DO SAME USING A `virtualenv` using python 3.7+\*

3. Now run the main function in the source code `python main.py` to start the proctoring system.
4. First register yourself on the first screen by pressing `r` to capture you images for 5 times.
5. Now monitoring will begin and will end after you press the `esc` key .
6. The reports generated will be stored in the `./results` folder.
