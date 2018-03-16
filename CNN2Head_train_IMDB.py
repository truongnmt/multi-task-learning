import CNN2Head_input
import os
import tensorflow as tf
import numpy as np
import BKNetStyle as BKNetStyle
from const import *

''' PREPARE DATA '''
smile_train, smile_test = CNN2Head_input.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = CNN2Head_input.getEmotionImage()
gender_train, gender_test = CNN2Head_input.getImdbImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_, mask = BKNetStyle.Input()

    y_smile_conv, y_emotion_conv, y_gender_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)

    smile_loss, emotion_loss, gender_loss, l2_loss, loss = BKNetStyle.selective_svm_loss(y_smile_conv, y_emotion_conv,
                                                                                         y_gender_conv, y_, mask)

    train_step = BKNetStyle.train_op(loss, global_step)

    smile_mask = tf.get_collection('smile_mask')[0]
    emotion_mask = tf.get_collection('emotion_mask')[0]
    gender_mask = tf.get_collection('gender_mask')[0]
    y_smile = tf.get_collection('y_smile')[0]
    y_emotion = tf.get_collection('y_emotion')[0]
    y_gender = tf.get_collection('y_gender')[0]

    smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
    emotion_correct_prediction = tf.equal(tf.argmax(y_emotion_conv, 1), tf.argmax(y_emotion, 1))
    gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))

    smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
    emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32) * emotion_mask)
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)

    train_data = []
    # Mask: Smile -> 0, Emotion -> 1, Gender -> 2
    for i in range(len(smile_train) * 50):
        img = (smile_train[i % 3000][0] - 128) / 255.0
        label = smile_train[i % 3000][1]
        train_data.append((img, one_hot(label, 7), 0.0))
    for i in range(len(emotion_train) * 5):
        train_data.append((emotion_train[i % len(emotion_train)][0], emotion_train[i % len(emotion_train)][1], 1.0))
    for i in range(len(gender_train)):
        img = (gender_train[i][0] - 128) / 255.0
        label = (int)(gender_train[i][1])
        train_data.append((img, one_hot(label, 7), 2.0))

    saver = tf.train.Saver()

    if not os.path.isfile(SAVE_FOLDER + 'model2.ckpt.index'):
        print('Create new model')
        sess.run(tf.global_variables_initializer())
        print('OK')
    else:
        print('Restoring existed model')
        saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./summary/")

    learning_rate = tf.get_collection('learning_rate')[0]

    current_epoch = (int)(global_step.eval() / (len(train_data) // BATCH_SIZE))
    for epoch in range(current_epoch + 1, NUM_EPOCHS):
        print('Epoch:', str(epoch))
        np.random.shuffle(train_data)
        train_img = []
        train_label = []
        train_mask = []

        for i in range(len(train_data)):
            train_img.append(train_data[i][0])
            train_label.append(train_data[i][1])
            train_mask.append(train_data[i][2])

        number_batch = len(train_data) // BATCH_SIZE

        avg_ttl = []
        avg_rgl = []
        avg_smile_loss = []
        avg_emotion_loss = []
        avg_gender_loss = []

        smile_nb_true_pred = 0
        emotion_nb_true_pred = 0
        gender_nb_true_pred = 0

        smile_nb_train = 0
        emotion_nb_train = 0
        gender_nb_train = 0
        print("Learning rate: %f" % learning_rate.eval())
        for batch in range(number_batch):
            print('Training on batch ', str(batch + 1), '/', str(number_batch), end='\r')
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(train_data))
            batch_img = np.asarray(train_img[top:bot])
            batch_label = np.asarray(train_label[top:bot])
            batch_mask = np.asarray(train_mask[top:bot])

            for i in range(BATCH_SIZE):
                if batch_mask[i] == 0.0:
                    smile_nb_train += 1
                else:
                    if batch_mask[i] == 1.0:
                        emotion_nb_train += 1
                    else:
                        gender_nb_train += 1

            batch_img = CNN2Head_input.augmentation(batch_img, 48)

            ttl, sml, eml, gel, l2l, _ = sess.run([loss, smile_loss, emotion_loss, gender_loss, l2_loss, train_step],
                                                  feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                             phase_train: True,
                                                             keep_prob: 0.5})

            smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                       phase_train: True,
                                                                       keep_prob: 0.5})

            emotion_nb_true_pred += sess.run(emotion_true_pred,
                                             feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                        phase_train: True,
                                                        keep_prob: 0.5})

            gender_nb_true_pred += sess.run(gender_true_pred,
                                            feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                       phase_train: True,
                                                       keep_prob: 0.5})

            '''--------------------------------------- DEBUG -----------------------------------------------------'''
            '''
            sm_mask, em_mask, ge_mask = sess.run([smile_mask, emotion_mask, gender_mask],
                                                 feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                            phase_train: True,
                                                            keep_prob: 0.5})
            print('Smile mask: ', sm_mask)
            print('Emotion mask', em_mask)
            print('Gender mask', ge_mask)
            print('Batch mask', batch_mask)

            y_true_sm, y_true_em, y_true_ge = sess.run([y_smile, y_emotion, y_gender],
                                                       feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                  phase_train: True,
                                                                  keep_prob: 0.5})
            print('Smile label', y_true_sm)
            print('Emotion label', y_true_em)
            print('Gender label', y_true_ge)
            print('Batch label', batch_label)

            y_conv_sm, y_conv_em, y_conv_ge = sess.run([y_smile_conv, y_emotion_conv, y_gender_conv],
                                                       feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                  phase_train: True,
                                                                  keep_prob: 0.5})
            print('Smile conv', y_conv_sm)
            print('Emotion conv', y_conv_em)
            print('Gender conv', y_conv_ge)
            '''
            '''---------------------------------- END OF DEBUG ----------------------------------------------------'''

            avg_ttl.append(ttl)
            avg_smile_loss.append(sml)
            avg_emotion_loss.append(eml)
            avg_gender_loss.append(gel)
            avg_rgl.append(l2l)

        smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train
        emotion_train_accuracy = emotion_nb_true_pred * 1.0 / emotion_nb_train
        gender_train_accuracy = gender_nb_true_pred * 1.0 / gender_nb_train

        avg_smile_loss = np.average(avg_smile_loss)
        avg_emotion_loss = np.average(avg_emotion_loss)
        avg_gender_loss = np.average(avg_gender_loss)
        avg_rgl = np.average(avg_rgl)
        avg_ttl = np.average(avg_ttl)

        summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl})
        writer.add_summary(summary, global_step=epoch)

        print('\n')

        print('Smile task train accuracy: ' + str(smile_train_accuracy * 100))
        print('Emotion task train accuracy: ' + str(emotion_train_accuracy * 100))
        print('Gender task train accuracy: ' + str(gender_train_accuracy * 100))
        print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
        print('Smile loss: ' + str(avg_smile_loss))
        print('Emotion loss: ' + str(avg_emotion_loss))
        print('Gender loss: ' + str(avg_gender_loss))

        # saver.save(sess, SAVE_FOLDER + 'model.ckpt')
