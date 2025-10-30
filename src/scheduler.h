#pragma once

#include "../common.h"

class Scheduler {
private:
    int BUFFER_STAGE; // 缓冲区阶段
    int CONSUMER_WARP_NUM; // 消费者WARP数量
    int PRODUCER_WARP_GROUP_NUM; // 生产者WARP组数量
    int SM_NUM; // SM数量

    // 私有构造函数，防止外部实例化
    Scheduler() : BUFFER_STAGE(0), CONSUMER_WARP_NUM(0), PRODUCER_WARP_GROUP_NUM(0), SM_NUM(0) {}

    // 禁止拷贝构造和赋值
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

public:
    static Scheduler& getInstance() {
        static Scheduler instance;
        return instance;
    }

    // 可以根据需要添加 Buffer、Consumer、Producer 设置/获取函数
    void setBufferStage(int val) { BUFFER_STAGE = val; }
    int getBufferStage() const { return BUFFER_STAGE; }

    void setConsumerWarpNum(int val) { CONSUMER_WARP_NUM = val; }
    int getConsumerWarpNum() const { return CONSUMER_WARP_NUM; }

    // 统一设置所有参数的接口
    void configure(int buffer_stage, int consumer_warp_num, int producer_warp_group_num, int sm_num) {
        BUFFER_STAGE = buffer_stage;
        CONSUMER_WARP_NUM = consumer_warp_num;
        PRODUCER_WARP_GROUP_NUM = producer_warp_group_num;
        SM_NUM = sm_num;
    }

    // 获取SM数量
    int getSMNum() const { return SM_NUM; }
};