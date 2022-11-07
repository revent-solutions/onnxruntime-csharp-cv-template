using Microsoft.ML.OnnxRuntime;
using OnnxRuntime.ResNet.Template.utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxRuntime.ResNet.Template
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Read paths
            string modelFilePath = @"model\u2net.onnx";
            string imageFilePath = @"data\test.jpg";

            var input = ImageHelper.GetImageTensorFromPath(imageFilePath);
            var result = ModelHelper.GetPredictions(input, modelFilePath);

            // Print results to console
            Console.WriteLine("--------------------------------------------------------------");
        }
    }
}
