import pynvml


# 字节数转GB
def bytes_to_gb(sizes):
    sizes = round(sizes / (1024 ** 3), 2)
    return f"{sizes} GB"


def gpu():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount(),  # 显卡数量
    if len(gpu_count) > 1:  # 如果有多张显卡
        for i in range(len(gpu_count)):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle=handle)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle=handle)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 显卡句柄（只有一张显卡）
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # 显卡名称
    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显卡内存信息
    data = dict(
        gpu_name=gpu_name.decode("utf-8"),
        gpu_count=len(gpu_count),
        gpu_memory_total=bytes_to_gb(gpu_memory.total),
        gpu_memory_used=bytes_to_gb(gpu_memory.used),
        gpu_memory_free=bytes_to_gb(gpu_memory.free),
    )
    return data

if __name__ == '__main__':
    gpu()
    1111
    3333
