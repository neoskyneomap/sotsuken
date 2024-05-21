# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
import tensorflow as tf
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
#from evaluate_best_player import evaluate_best_player

# GPUの設定を確認する
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 設定により、必要なメモリのみを確保するように設定
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # メモリ設定はプログラム開始前に行う必要があるため、エラーが発生する可能性があります。
        print(e)
        
# デュアルネットワークの作成
dual_network()

for i in range(10):
    print('Train',i,'====================')
    # セルフプレイ部
    self_play()

    # パラメータ更新部
    train_network()

    # 新パラメータ評価部
    update_best_player = evaluate_network()

    # ベストプレイヤーの評価
    # if update_best_player:
        # evaluate_best_player()