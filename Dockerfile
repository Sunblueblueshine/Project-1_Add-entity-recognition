FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends\
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .


# 安装uv
RUN pip install uv

#创建虚拟环境
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# 增加 uv 下载超时和重试
ENV UV_HTTP_TIMEOUT=300
ENV UV_HTTP_RETRIES=10

# 通过uv安装所有的依赖（带缓存）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

# 创建必要的目录
RUN mkdir -p tmp_files

# 暴露端口
EXPOSE 8020

# 镜像运行时执行的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]