import cv2
import playsound
from threading import Thread
import numpy as np
import onnx
import onnxruntime
from PIL import Image
import multiprocessing
import os
import time
class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath),providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] )
        assert len(self.session.get_inputs()) == 1
        print(self.session.get_inputs()[0].shape[2:])
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True
    def from_cv2(self,img):
        # You may need to convert the color.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    def predict(self, img):
        img = img.resize(self.input_shape)
        input_array = np.array(img, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

def main():
    model = Model(rf"{os.getcwd()}\490fadb34c604b9089c86fd9730fada3.ONNX\model.onnx")

    frame = cv2.imread(r"C:\Users\Mumuji\Pictures\Camera Roll\WIN_20220511_14_50_32_Pro.jpg")
    print(frame)
    violence = []
    outs = model.predict(model.from_cv2(frame))
    cv2.imshow("cheesiest",frame)
    prob = outs["loss"][0]['Violence']
    print(type(prob))
    
    frame = cv2.putText(frame,f"{(prob)*100}%", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255),3)
    #print(sum(violence)/len(violence))
    cv2.imshow("frame",frame)
    cv2.imwrite("outest.png",frame)
    cv2.waitKey()
    
    
if __name__ == "__main__":
    main()