问题总结     
  我找到了 Lorentz 模型在 humerus_left 类别 CD 指标异常的根本原因：   
  关键发现： 
  1. 中位数 CD 正常：
    - Baseline: 5.43mm
    - Lorentz: 5.35mm                  
    - Poincare: 5.24mm 
  2. 平均 CD 被极端异常值拉高：
    - Lorentz 最大值：1372mm（样本 BDMAP_00004349）
    - 该样本中 Lorentz 预测 humerus_left 在 z=184-187，而 GT 在 z=2-18
    - 预测位置偏差 707mm！
  3. 根本原因：
    - 不同的 target_size：Baseline (128×128×256) vs Lorentz/Poincare (160×160×256)
    - Lorentz 在某些样本中错误地将完全不相关的体素预测为 humerus_left                                                                       
    - 这些错误预测往往远离真实位置，导致极大的 CD 值                                                                                        
                                                                                                                                            
  建议解决方案：  
  1. 重新统一评测：使用相同的 target_size 重新生成所有模型的预测结果                                                                        
  2. 检查 Lorentz 训练/推理代码：查看为什么会在远离 GT 的位置预测 humerus_left                                                              
  3. 临时方案：使用中位数 CD 而不是平均 CD 来评估，因为中位数更能反映大多数样本的真实表现                                                   
                                                                                                                                            
  需要我帮你检查 Lorentz 模型的训练或推理代码吗？