import tensorflow as tf


#总结一下这版实现：
#输入userid，itemid，history item id list
#wide侧：用itemid embedding和history item id list的embedding，抽取3维数据(1,65,128)，分别做elementwise乘法，得到3维独立向量，
#concat到一起得到[batch_size, 3]的向量，过一个tf.dense层输出1维，这个过程就实现了LR，相当于y=w1x1+w2x2+w3x3，模型在学习w1,w2,w3
#deep侧：itemid和history item id list作为输入，过几层fc，最后输出1维
#wide侧1维+deep侧一维+bias一维，然后过sigmoid
#实现上有个要点，wide侧和deep侧是共享embedding的

#测试集的生成方式比较有意思，用userid和timestamp排序后，同一个用户最后一条样本用来做测试集，如果只有一条用户行为的，都拿去做测试集了
#因为用timestamp排了序，这样能保证模型不会看到未来的样本

#几个疑问：
#1. 为啥废弃了userid这个特征
#2. 为啥wide侧的实现，是这么做特征交叉的

#实际项目中，可以这么做：
#1. wide侧：user侧设计一堆特征，item侧设计一堆特征，基于这两堆特征设计一些组合特征，用这些特征过一个tf.dense层，输出1维

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list):

    #placeholder always cooprate with Session feed_list
    self.u = tf.placeholder(tf.int32, [None,]) # [B]  B代表batch_size
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    #use self.u as a key to lookup its embedding. self.u has already been trans to a feature sign
    #这个feature sign不是全局唯一的，而是在这个slot下处理成了从0开始的唯一的索引
    #user_emb_w是模型训练的参数
    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    #从cate_list中获取索引为i的cate
    ic = tf.gather(cate_list, self.i)
    #item embedding
    i_emb = tf.concat(values = [
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
    #item bias
    i_b = tf.gather(item_b, self.i)

    #for evaluation
    jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
    j_b = tf.gather(item_b, self.j)

    #从cate_list中获取索引为hist_i（一个数组）的sub cate list，注意这里axis是2，对应获取数组形式的index
    hc = tf.gather(cate_list, self.hist_i)
    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

    #-- sum begin --------
    # mask the zero padding part
    # batch_size中每个样本，其hist len都不一样，这里会把列数扩充成max_len:tf.shape(h_emb)[1]，即把每个样本对应长度个元素赋值为1，其他的填充成0
    # h_emb是一个三维数组，第一维是batch_size, 第二维是max hist len，第三维是item和cate的embedding concat后的长度，在这里是64+64=128
    # sequence_mask用法：https://blog.csdn.net/xinjieyuan/article/details/95760679
    mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
    #-1表示最后一维，即往后扩展一维
    #expand_dims:https://blog.csdn.net/duanlianvip/article/details/96448393
    mask = tf.expand_dims(mask, -1) # [B, T, 1]
    #对mask进行扩展，第一第二维都不扩展，第三维扩展成embedding的长度
    #tf.tile:https://blog.csdn.net/tsyccnh/article/details/82459859
    mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]
    #embedding和mask相乘，目的是把没有对应hist的postion上的embedding置为0，网上有一段解释：
    #sequence mask的应用场景主要是在填充计算时候使用，比如你把没有单词的位置填充了0，如果纳入了前向传播计算，影响了最终经验损失函数的结果。
    #那么我们如果通过tf.sequence_mask得到的mask张量，与损失函数结果进行对照相乘，可以去掉无用的损失值，保证了计算的准确性。
    h_emb *= mask # [B, T, H]
    hist = h_emb
    #axis=0表示多个样本，axis=1代表的是每个用户的多个hist，axis=2表示每个hist的embedding
    #则效果是对每个样本，hist len个embedding进行求和，操作完变为两维，reduce了一维，第一维是batch_size，第二维是embedding加和后的长度，
    #在这里是128，即hist shape为[batch_size, 128]
    #reduce_sum:https://www.zhihu.com/question/51325408
    hist = tf.reduce_sum(hist, 1)
    #self.sl为[batch_size]，扩展为[batch_size, 128]，每个元素都是一样的，举例self.sl为[3,5,7,9]，
    #则扩展后为[[3,3,...,3],[5,5,...,5],..,[9,9,...,9]]，中间省略号一共128个。实现的效果是对每个样本，
    #用hist平均除以每个样本的hist len，上述一系列操作就是在做avg pooling
    hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl,1), [1,128]), tf.float32))
    print h_emb.get_shape().as_list()
    #-- sum end ---------
    
    hist = tf.layers.batch_normalization(inputs = hist)
    hist = tf.reshape(hist, [-1, hidden_units])
    hist = tf.layers.dense(hist, hidden_units)

    #这里为啥要做这个赋值？没看懂，看起来user本身的embedding被抹掉了，替换成了行为序列hist产生的embedding
    u_emb = hist
    #-- fcn begin -------
    din_i = tf.concat([u_emb, i_emb], axis=-1)
    din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
    d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
    d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
    # wide part
    #这里如何理解？看起来是把u_emb的第1维，第64+1维，第128维，和i_emb的对应维相乘，相当于选了几个维度进行特征交叉
    d_layer_wide_i = tf.concat([tf.gather(u_emb, [0], axis=-1) * tf.gather(i_emb, [0], axis=-1), tf.gather(u_emb, [-1], axis=-1) * tf.gather(i_emb, [-1], axis=-1),
                     tf.gather(u_emb, [hidden_units // 2], axis=-1) * tf.gather(i_emb, [hidden_units // 2], axis=-1)], axis=-1)
    d_layer_wide_i = tf.layers.dense(d_layer_wide_i, 1, activation=None, name='f_wide')

    #j相关的都是for evaluation
    din_j = tf.concat([u_emb, j_emb], axis=-1)
    din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
    d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_wide_j = tf.concat([tf.gather(u_emb, [0], axis=-1) * tf.gather(j_emb, [0], axis=-1), tf.gather(u_emb, [-1], axis=-1) * tf.gather(j_emb, [-1], axis=-1),
                     tf.gather(u_emb, [hidden_units // 2], axis=-1) * tf.gather(j_emb, [hidden_units // 2], axis=-1)], axis=-1)
    d_layer_wide_j = tf.layers.dense(d_layer_wide_j, 1, activation=None, name='f_wide', reuse=True)

    d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
    d_layer_wide_i = tf.reshape(d_layer_wide_i, [-1])
    d_layer_wide_j = tf.reshape(d_layer_wide_j, [-1])
    x = i_b - j_b + d_layer_3_i - d_layer_3_j + d_layer_wide_i - d_layer_wide_j # [B]

    #这个是真正要用来计算loss的logits
    self.logits = i_b + d_layer_3_i + d_layer_wide_i

    #u_emb扩充前[batch_size, 128]，扩充后[batch_size, item_count, 128]
    u_emb_all = tf.expand_dims(u_emb, 1)
    u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
    # logits for all item:
    # all_emb:[item_count, 64+64=128]
    all_emb = tf.concat([
        item_emb_w,
        #这里不用cate_emb_w，而是用tf.nn.embedding_lookup(cate_emb_w, cate_list)的原因是：不同item的cate可能会重复，比如cate_emb_w存储了20个cate的
        #embedding，但cate_list可能有30个cate，其中10个是重复的，这里需要把30个cate的embedding都取出来
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    all_emb = tf.expand_dims(all_emb, 0)
    #all_emb:[512, item_count, 128]，512是test_batch_size，512属于硬编码了
    all_emb = tf.tile(all_emb, [512, 1, 1])
    din_all = tf.concat([u_emb_all, all_emb], axis=-1)
    din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
    d_layer_wide_all = tf.concat([tf.gather(u_emb_all, [0], axis=-1) * tf.gather(all_emb, [0], axis=-1), tf.gather(u_emb_all, [-1], axis=-1) * tf.gather(all_emb, [-1], axis=-1), tf.gather(u_emb_all, [hidden_units // 2], axis=-1) * tf.gather(all_emb, [hidden_units // 2], axis=-1)], axis=-1)
    d_layer_wide_all = tf.layers.dense(d_layer_wide_all, 1, activation=None, name='f_wide', reuse=True)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
    d_layer_wide_all = tf.reshape(d_layer_wide_all, [-1, item_count])
    self.logits_all = tf.sigmoid(item_b + d_layer_3_all + d_layer_wide_all)
    #-- fcn end -------

    #正负样本pair对中，正样本打分大于负样本打分的数量
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_b + d_layer_3_i + d_layer_wide_i)
    self.score_j = tf.sigmoid(j_b + d_layer_3_j + d_layer_wide_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
    print self.p_and_n.get_shape().as_list()


    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    regulation_rate = 0.0
    self.loss = tf.reduce_mean(
        #这个函数本身会对logits计算sigmoid后，再计算loss
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        )

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    #每次session run的时候，根据fetch list构建数据流图，feed_dict提供给placeholder
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss

  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    return u_auc, socre_p_and_n

  def test(self, sess, uid, hist_i, sl):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        self.hist_i: hist_i,
        self.sl: sl,
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

