import os
import torch
from django.shortcuts import render
from django.views import View
from django.conf import settings
from .apps import ModelConfig
from torch.utils.data import DataLoader
from .yolo.utils.datasets import *
from .yolo.models import *
from .yolo.utils.utils import *
from .yolo.utils.datasets import *
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np

# Create your views here.

class Index(View):
    template_name = 'index.html'
    def get(self,request):
        return render(request, self.template_name,{})

    def post(self,request):
        image = request.FILES['img']
        ext = image.name.split('.')[-1]
        for filename in os.listdir(settings.IMAGE_ROOT):
            os.remove(os.path.join(settings.IMAGE_ROOT,filename))
        with open(os.path.join(settings.IMAGE_ROOT,'img.'+str(ext)), 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)
        dataloader = DataLoader(ImageFolder(settings.IMAGE_ROOT),
                                batch_size=1,
                                shuffle=False)

        classes = load_classes( os.path.join(settings.MODELS_ROOT,'classes.names'))  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = ModelConfig.model(input_imgs)
                detections = non_max_suppression(detections, conf_thres=0.5, nms_thres=0.4)


            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            print("(%d) Image: '%s'" % (img_i, path))

            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, 416, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig("pred.png", bbox_inches="tight", pad_inches=0.0)
            plt.close()

        return render(request, self.template_name,{})