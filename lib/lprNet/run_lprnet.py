import sys

sys.path.append(".")
from lib.lprNet.LPRNetVN import *

global_step = tf.Variable(0, trainable=False)
logits, inputs, seq_len = get_train_model(num_channels, label_len, 1, img_size)
logits = tf.transpose(logits, (1, 0, 2))
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

def extract_license_number(img):

    image = np.zeros([1, 25, 100, 3])
    img = cv2.resize(img, (100, 25), interpolation=cv2.INTER_CUBIC)

    image[0, ...] = img
    image = np.transpose(image, axes=[0,2,1,3])

    seq_len_1 = np.ones(1) * 24



    test_feed = {inputs: image,
                seq_len: seq_len_1}
    dd = session.run(decoded[0], test_feed)
    detected_list = decode_sparse_tensor(dd)

    return ("".join(detected_list[0]).strip())


init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
saver.restore(session, './model/ocr_dec10/LPRNetVN.ckpt-290000')

