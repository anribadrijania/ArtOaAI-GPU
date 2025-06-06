# ArtOa
 AI project for ArtOa GPU model.

##### Start with:
1. Install Python 3.13.3
2. Install Cuda Toolkit from https://developer.nvidia.com/cuda-downloads (additional details in https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages)
3. git clone (repo)
4. cd (repo)
5. python3.13 -m venv venv
6. .\venv\Scripts\Activate.ps1
7. pip install -r requirements.txt
8. add dependencies in ArtOa-AI-Dependencies-GPU.rar to the pipeline folder (pretrained folder, remover_v2_GPU.pth, maskrcnn_v2.pth)
9. create the .env file in the pipeline folder and define your Azure endpoint, Azure OpenAI API key and application API key.
10. cd pipeline
11. type <uvicorn app:app --reload> for debugging mode or <gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 2 --bind 127.0.0.1:8000> for production (define host, port and number of workers by you preferences
