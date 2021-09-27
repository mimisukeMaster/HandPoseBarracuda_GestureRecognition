using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;  // import必須
using System.Collections.Generic;
using System.Linq;

namespace MediaPipe.HandPose
{
    public class Classification : MonoBehaviour
    {
        //  Barracuda 推論用
        public NNModel modelAsset;
        private Model m_RuntimeModel;
        private IWorker m_worker;

        //public RenderTexture targetTexture;
        public Text targetText;
        [SerializeField] ResourceSet _resources = null;
        HandPipeline _pipelineCla;
        public WebcamInput _webcam = null;
        // Start is called before the first frame update
        void Start()
        {
            m_RuntimeModel = ModelLoader.Load(modelAsset);

            var workerType = WorkerFactory.Type.Compute; // GPUで実行する場合はこちらを利用
                                                         // var workerType = WorkerFactory.Type.CSharp;  // CPUで実行する場合はこちらを利用

            m_worker = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);

            _pipelineCla = new HandPipeline(_resources);
        }

        private void Update()
        {
            _pipelineCla.UseAsyncReadback = true;
            _pipelineCla.ProcessImage(_webcam.Texture);

            //1 List型でXY別々に値入れるList作成,然るべき値代入
            List<float> KeyPointX = new List<float>();
            List<float> KeyPointY = new List<float>();

            for (int i = 0; i < HandPipeline.KeyPointCount; i++)
            {
                KeyPointX.Add(_pipelineCla.GetKeyPoint(i).x);
                KeyPointY.Add(_pipelineCla.GetKeyPoint(i).y);
            }

            
            //2 0番目(根本)の手Pointの座標をXY共にとっておく
            float baseX = KeyPointX[0];
            float baseY = KeyPointY[0];

            //3 手の座標全てに対し0番目の手の座標との差を入れ,上書きする yは正負が逆なので *-1 する(kazuhito00方式にあわせる)
            for (int j = 0; j < HandPipeline.KeyPointCount; j++)
            {
                KeyPointX[j] = KeyPointX[j] - baseX;
                KeyPointY[j] = (KeyPointY[j] - baseY) * -1;
            }
            
            //4 別々になっているX,Yの手の座標値Listを一つのリストにする
            List<float> pointVector = new List<float>();

            for(int m = 0; m < HandPipeline.KeyPointCount; m++)
            {
                pointVector.Add(KeyPointX[m]);
                pointVector.Add(KeyPointY[m]);
            }
            
            //5 合成したListから最大値を見つける  一個一個の要素に対し絶対値をとりその上最大値を見つける
            List<float> AbsKeyPointXY = new List<float>();

            for(int k = 0; k < pointVector.Count; k++)
            {
                AbsKeyPointXY.Add(Mathf.Abs(pointVector[k]));
            }
            float XYmax = AbsKeyPointXY.Max();

            //6 各々のListの各要素に対し, 5 で見つけた最大値で割り正規化  それをまたListに上書きする
            for(int l = 0; l < pointVector.Count; l++)
            {
                pointVector[l] = pointVector[l] / XYmax;
            }
            
            for(int i = 0; i < pointVector.Count / 2; i++)
            {
                Debug.Log("X[" + i + "]: " + pointVector[i * 2 ] + "  Y[" + i + "]: " + pointVector[i * 2 + 1] + "\n"  );
            }


            //listを配列に変換 これを推論に持っていく
            float[] pointVectorArray = pointVector.ToArray();
            string z = "";
            Tensor handInput = new Tensor(1, 1, 1, 42,pointVectorArray,z);

            Inference(handInput);
            handInput.Dispose();
        }

        private void Inference(Tensor input)
        {
            //Debug.Log("inputは　" + input);
            m_worker.Execute(input);
            Tensor output = m_worker.PeekOutput();

            var outputArray = output.ToReadOnlyArray();
            int maxIndex = 0;
            float max = 0;
            for (int i = 0; i < outputArray.Length; i++)
            {
                if (max < outputArray[i])
                {
                    max = outputArray[i];
                    maxIndex = i;
                }
            }

            targetText.text = Library.getImageNetSynset()[maxIndex];
            //Debug.Log("推論した結果　" + output);
            output.Dispose();   //各ステップごとにTensorは破棄する必要がある(メモリリーク回避のため)
        }

        private void OnDestroy()
        {
            m_worker.Dispose(); //終了時に破棄する
            _pipelineCla.Dispose();
        }

    }
}