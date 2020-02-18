from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt


def imshow2(img, pred):
    
    img = TF.to_pil_image(img, mode='L')
    pred = TF.to_pil_image(pred, mode='L')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(img)
    #plt.title('image')
    
    plt.subplot(122)
    plt.imshow(pred)
    #plt.title('pred')
    
    plt.show()    
    
def imshow3(img, label, pred):
    
    img = TF.to_pil_image(img, mode='L')
    label = TF.to_pil_image(label, mode='L')
    pred = TF.to_pil_image(pred, mode='L')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(132)
    plt.imshow(label)
    plt.title('label')
    
    plt.subplot(133)
    plt.imshow(pred)
    plt.title('pred')
    
    plt.show()   
    
def imshow4(img1, label1, img2, label2):
    
    img1 = TF.to_pil_image(img1, mode='L')
    label1 = TF.to_pil_image(label1, mode='L')
    img2 = TF.to_pil_image(img2, mode='L')
    label2 = TF.to_pil_image(label2, mode='L')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(img1)
    #plt.title('image')
    
    plt.subplot(142)
    plt.imshow(label1)
    #plt.title('pred')
    
    plt.subplot(143)
    plt.imshow(img2)
    #plt.title('image')
    
    plt.subplot(144)
    plt.imshow(label2)
    
    plt.show()   
    
def show_plot(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.grid()
    plt.show()