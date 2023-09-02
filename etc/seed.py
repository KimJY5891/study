def set_global_seed(seed_value=42):
    """
    주요 라이브러리들의 랜덤 시드 값을 고정하는 함수.
    """
    # Python 내장 random 라이브러리
    random.seed(seed_value)

    # NumPy
    np.random.seed(seed_value)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
    except ImportError:
        pass  # TensorFlow가 설치되어 있지 않으면 건너뜁니다.

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch가 설치되어 있지 않으면 건너뜁니다.

    # OS (시스템 콜에 영향을 주는 랜덤 시드)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
set_global_seed(128)
