import CNN2Head_input
import tensorflow as tf
import numpy as np

SAVE_FOLDER = 'D:/LeTranBaoCuong/CNN2Head/save/current'

_, public_test_data, private_test_data = CNN2Head_input.getEmotionImage()
_, smile_test_data =  CNN2Head_input.getSmileImage()

def eval_smile_public_test_emotion(nbof_crop):
    nbof_smile = len(smile_test_data)
    nbof_emotion = len(public_test_data)

    nbof_true_emotion = 0
    nbof_true_smile = 0

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(SAVE_FOLDER + '/model.ckpt.meta')
    saver.restore(sess, SAVE_FOLDER + "/model.ckpt")
    x_smile = tf.get_collection('x_smile')[0]
    x_emotion = tf.get_collection('x_emotion')[0]
    keep_prob_smile_fc1 = tf.get_collection('keep_prob_smile_fc1')[0]
    keep_prob_emotion_fc1 = tf.get_collection('keep_prob_emotion_fc1')[0]
    keep_prob_smile_fc2 = tf.get_collection('keep_prob_smile_fc2')[0]
    keep_prob_emotion_fc2 = tf.get_collection('keep_prob_emotion_fc2')[0]
    y_smile_conv = tf.get_collection('y_smile_conv')[0]
    y_emotion_conv = tf.get_collection('y_emotion_conv')[0]
    is_training = tf.get_collection('is_training')[0]

    for i in range(nbof_emotion):
        emotion = np.zeros([1,48,48,1])
        emotion[0] = public_test_data[i][0]
        emotion_label = np.argmax(public_test_data[i][1])
        smile = np.zeros([1,96,96,1])
        smile[0] = smile_test_data[i % 1000][0]
        smile_label = smile_test_data[i % 1000][1]
        y_emotion_pred = np.zeros([7])
        y_smile_pred = np.zeros([2])

        for _ in range(nbof_crop):
            x_emotion_ = CNN2Head_input.random_crop(emotion, (48, 48), 10)
            x_smile_ = CNN2Head_input.random_crop(smile,(96, 96), 10)
            y1 = y_emotion_conv.eval(feed_dict= {x_smile: x_smile_,
                                                 x_emotion: x_emotion_,
                                                 keep_prob_smile_fc1: 1,
                                                 keep_prob_smile_fc2: 1,
                                                 keep_prob_emotion_fc1: 1,
                                                 keep_prob_emotion_fc2: 1,
                                                 is_training: False})
            y2 = y_smile_conv.eval(feed_dict={x_smile: x_smile_,
                                                x_emotion: x_emotion_,
                                                keep_prob_smile_fc1: 1,
                                                keep_prob_smile_fc2: 1,
                                                keep_prob_emotion_fc1: 1,
                                                keep_prob_emotion_fc2: 1,
                                                is_training: False})
            y_emotion_pred += y1[0]
            y_smile_pred += y2[0]

        predict_emotion = np.argmax(y_emotion_pred)
        predict_smile = np.argmax(y_smile_pred)

        if (predict_emotion == emotion_label):
            nbof_true_emotion += 1
        if (predict_smile == smile_label) & (i < 1000):
            nbof_true_smile += 1
    return nbof_true_smile * 100.0 / nbof_smile, nbof_true_emotion * 100.0 / nbof_emotion


def eval_smile_private_test_emotion(nbof_crop):
    nbof_smile = len(smile_test_data)
    nbof_emotion = len(private_test_data)

    nbof_true_emotion = 0
    nbof_true_smile = 0

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(SAVE_FOLDER + '/model.ckpt.meta')
    saver.restore(sess, SAVE_FOLDER + "/model.ckpt")
    x_smile = tf.get_collection('x_smile')[0]
    x_emotion = tf.get_collection('x_emotion')[0]
    keep_prob_smile_fc1 = tf.get_collection('keep_prob_smile_fc1')[0]
    keep_prob_emotion_fc1 = tf.get_collection('keep_prob_emotion_fc1')[0]
    keep_prob_smile_fc2 = tf.get_collection('keep_prob_smile_fc2')[0]
    keep_prob_emotion_fc2 = tf.get_collection('keep_prob_emotion_fc2')[0]
    y_smile_conv = tf.get_collection('y_smile_conv')[0]
    y_emotion_conv = tf.get_collection('y_emotion_conv')[0]
    is_training = tf.get_collection('is_training')[0]

    for i in range(nbof_emotion):
        emotion = np.zeros([1,48,48,1])
        emotion[0] = private_test_data[i][0]
        emotion_label = np.argmax(private_test_data[i][1])
        smile = np.zeros([1,96,96,1])
        smile[0] = smile_test_data[i % 1000][0]
        smile_label = smile_test_data[i % 1000][1]
        y_emotion_pred = np.zeros([7])
        y_smile_pred = np.zeros([2])

        for _ in range(nbof_crop):
            x_emotion_ = CNN2Head_input.random_crop(emotion, (48, 48), 10)
            x_smile_ = CNN2Head_input.random_crop(smile,(96, 96), 10)
            y1 = y_emotion_conv.eval(feed_dict= {x_smile: x_smile_,
                                                 x_emotion: x_emotion_,
                                                 keep_prob_smile_fc1: 1,
                                                 keep_prob_smile_fc2: 1,
                                                 keep_prob_emotion_fc1: 1,
                                                 keep_prob_emotion_fc2: 1,
                                                 is_training: False})
            y2 = y_smile_conv.eval(feed_dict={x_smile: x_smile_,
                                                x_emotion: x_emotion_,
                                                keep_prob_smile_fc1: 1,
                                                keep_prob_smile_fc2: 1,
                                                keep_prob_emotion_fc1: 1,
                                                keep_prob_emotion_fc2: 1,
                                                is_training: False})
            y_emotion_pred += y1[0]
            y_smile_pred += y2[0]

        predict_emotion = np.argmax(y_emotion_pred)
        predict_smile = np.argmax(y_smile_pred)

        if (predict_emotion == emotion_label):
            nbof_true_emotion += 1
        if (predict_smile == smile_label) & (i < 1000):
            nbof_true_smile += 1
    return nbof_true_smile * 100.0 / nbof_smile, nbof_true_emotion * 100.0 / nbof_emotion


def evaluate(nbof_crop):
    print('Testing phase...............................')
    smile_acc, public_acc = eval_smile_public_test_emotion(nbof_crop)
    _, private_acc = eval_smile_private_test_emotion(nbof_crop)
    print('Smile test accuracy: ',str(smile_acc))
    print('Emotion public test accuracy: ', str(public_acc))
    print('Emotion private test accuracy: ', str(private_acc))

evaluate(10)



