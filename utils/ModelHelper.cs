using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace OnnxRuntime.ResNet.Template.utils
{
    public static class ModelHelper
    {
        public static Image<Rgb24> GetPredictions(Tensor<float> input, string modelFilePath)
        {
            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", input)
            };

            // Run inference
            var session = new InferenceSession(modelFilePath);
            var results = session.Run(inputs).First();

            int width = 224;
            int height = 224;

            var test = results.Value as DenseTensor<float>; 


            var result = new Image<Rgb24>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgb24(
                        (byte)Math.Clamp(test[ y, x], 0, 255),
                        (byte)Math.Clamp(test[y, x] , 0, 255),
                        (byte)Math.Clamp(test[ y, x] , 0, 255)
                    );
                }
            }
            result.Save("test.jpg");

            return result;
        }


    }
}
