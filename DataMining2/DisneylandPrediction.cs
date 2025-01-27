﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining2
{
    public class DisneylandPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint Prediction { get; set; }
        public float[] Probability { get; set; }
        public float[] Score { get; set; }
    }
}
