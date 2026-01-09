
//+------------------------------------------------------------------+
//|                                                  Atl4sBridge.mq5 |
//|                                  Copyright 2025, Atl4s Corp AI   |
//|                                             https://atl4s.ai     |
//+------------------------------------------------------------------+
#property copyright "Atl4s Corp AI"
#property link      "https://atl4s.ai"
#property version   "3.00"
#property description "AGI Ultra-Intelligent Bridge with Local ML, Adaptive Execution, and Smart Risk Management"
#property strict
#include <Trade\Trade.mqh>

// --- INPUTS ---
input int    InpPort = 5558; // User defined port
input int    MagicNumber = 1337;
input int    MaxSlippage = 5;
input double MaxDailyLoss = 5000.0;      // Maximum daily loss in account currency (Scaled for $8k)
input double MaxRiskPerTrade = 5.0;     // Maximum risk per trade (%) (Scaled for Aggressive)
input bool   EnableLocalML = true;       // Enable on-device ML
input bool   EnableAdaptiveExecution = true; // Enable adaptive execution

// --- GLOBALS ---
int socket_handle = INVALID_HANDLE;
string host = "127.0.0.1";
CTrade trade;

// ============================================================================
// LOCAL INTELLIGENCE (On-Device Analysis)
// ============================================================================

class CLocalIntelligence {
private:
    double market_state[];
    double decision_confidence;
    double last_prices[];
    int state_length;
    
    // Simple moving averages
    double sma_fast;
    double sma_slow;
    double rsi_value;
    
public:
    CLocalIntelligence() {
        state_length = 50;
        ArrayResize(market_state, state_length);
        ArrayResize(last_prices, state_length);
        decision_confidence = 0.0;
        sma_fast = 0;
        sma_slow = 0;
        rsi_value = 50;
    }
    
    void UpdateMarketState() {
        // Shift and add new price
        for(int i = state_length - 1; i > 0; i--) {
            last_prices[i] = last_prices[i-1];
        }
        last_prices[0] = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        
        // Calculate SMAs
        sma_fast = CalculateSMA(10);
        sma_slow = CalculateSMA(30);
        
        // Calculate RSI
        rsi_value = CalculateRSI(14);
        
        // Update market state vector
        market_state[0] = sma_fast;
        market_state[1] = sma_slow;
        market_state[2] = rsi_value;
        market_state[3] = CalculateVolatility(14);
        market_state[4] = (sma_fast > sma_slow) ? 1.0 : -1.0;
    }
    
    double CalculateSMA(int period) {
        if(period > state_length) return 0;
        double sum = 0;
        for(int i = 0; i < period; i++) {
            sum += last_prices[i];
        }
        return sum / period;
    }
    
    double CalculateRSI(int period) {
        if(period >= state_length) return 50;
        
        double gains = 0, losses = 0;
        for(int i = 0; i < period; i++) {
            double change = last_prices[i] - last_prices[i+1];
            if(change > 0) gains += change;
            else losses -= change;
        }
        
        if(losses == 0) return 100;
        double rs = gains / losses;
        return 100 - (100 / (1 + rs));
    }
    
    double CalculateVolatility(int period) {
        if(period >= state_length) return 0;
        
        double mean = 0;
        for(int i = 0; i < period; i++) mean += last_prices[i];
        mean /= period;
        
        double variance = 0;
        for(int i = 0; i < period; i++) {
            double diff = last_prices[i] - mean;
            variance += diff * diff;
        }
        
        return MathSqrt(variance / period);
    }
    
    string GenerateDecision() {
        UpdateMarketState();
        
        // Simple decision logic
        double trend_strength = (sma_fast - sma_slow) / sma_slow * 100;
        
        if(trend_strength > 0.1 && rsi_value < 70) {
            decision_confidence = MathMin(0.9, 0.5 + trend_strength);
            return "BUY";
        } else if(trend_strength < -0.1 && rsi_value > 30) {
            decision_confidence = MathMin(0.9, 0.5 - trend_strength);
            return "SELL";
        }
        
        decision_confidence = 0.3;
        return "WAIT";
    }
    
    double GetConfidence() { return decision_confidence; }
    double GetSMAFast() { return sma_fast; }
    double GetSMASlow() { return sma_slow; }
    double GetRSI() { return rsi_value; }
};

// ============================================================================
// ADAPTIVE EXECUTION
// ============================================================================

class CAdaptiveExecutor {
private:
    double spread_history[];
    int spread_idx;
    double avg_spread;
    double volatility_threshold;
    double optimal_execution_window;
    
public:
    CAdaptiveExecutor() {
        ArrayResize(spread_history, 100);
        spread_idx = 0;
        avg_spread = 0;
        volatility_threshold = 0.5;
        optimal_execution_window = 100; // ms
    }
    
    void Update() {
        double current_spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        spread_history[spread_idx] = current_spread;
        spread_idx = (spread_idx + 1) % 100;
        
        // Calculate average spread
        double sum = 0;
        for(int i = 0; i < 100; i++) sum += spread_history[i];
        avg_spread = sum / 100.0;
    }
    
    bool ShouldExecute(string action, double confidence) {
        Update();
        
        double current_spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        
        // Check spread conditions
        if(avg_spread > 0 && current_spread > avg_spread * 2.5) {
            Print("ADAPTIVE: Spread too high (", current_spread, " vs avg ", avg_spread, ")");
            return false;
        }
        
        // Check confidence threshold
        if(confidence < 0.5) {
            Print("ADAPTIVE: Confidence too low (", confidence, ")");
            return false;
        }
        
        return true;
    }
    
    double CalculateOptimalSize(double confidence, double risk_percent, double sl_distance) {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double risk_amount = balance * risk_percent / 100.0;
        
        // Adjust by confidence
        risk_amount *= confidence;
        
        // Calculate lot size based on SL distance
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        
        if(tick_value <= 0 || tick_size <= 0 || sl_distance <= 0) return 0.01;
        
        double lot_size = risk_amount / (sl_distance / tick_size * tick_value);
        
        // Clamp to symbol limits
        double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
        double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        lot_size = MathFloor(lot_size / step) * step;
        lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
        
        return lot_size;
    }
    
    double GetAvgSpread() { return avg_spread; }
};

// ============================================================================
// SMART ORDER MANAGEMENT
// ============================================================================

struct ShadowOrder {
    ulong ticket;
    double virtual_sl;
    double virtual_tp;
    double trailing_step;
    double breakeven_trigger;
    bool active;
};

class CSmartOrderManager {
private:
    ShadowOrder orders[];
    int max_orders;
    
public:
    CSmartOrderManager() {
        max_orders = 100;
        ArrayResize(orders, max_orders);
        for(int i = 0; i < max_orders; i++) orders[i].active = false;
    }
    
    void AddOrder(ulong ticket, double sl, double tp, double trail = 0, double be_trigger = 0) {
        for(int i = 0; i < max_orders; i++) {
            if(!orders[i].active) {
                orders[i].ticket = ticket;
                orders[i].virtual_sl = sl;
                orders[i].virtual_tp = tp;
                orders[i].trailing_step = trail;
                orders[i].breakeven_trigger = be_trigger;
                orders[i].active = true;
                Print("SMART ORDER: Added shadow order for ", ticket);
                return;
            }
        }
    }
    
    void ManageOrders() {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        
        for(int i = 0; i < max_orders; i++) {
            if(!orders[i].active) continue;
            
            if(!PositionSelectByTicket(orders[i].ticket)) {
                orders[i].active = false;
                continue;
            }
            
            long type = PositionGetInteger(POSITION_TYPE);
            double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
            double current_price = (type == POSITION_TYPE_BUY) ? bid : ask;
            double profit_pips = (type == POSITION_TYPE_BUY) ? 
                                 (current_price - open_price) : (open_price - current_price);
            
            // Breakeven logic
            if(orders[i].breakeven_trigger > 0 && profit_pips >= orders[i].breakeven_trigger) {
                if(orders[i].virtual_sl < open_price && type == POSITION_TYPE_BUY) {
                    orders[i].virtual_sl = open_price + SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 2;
                    Print("SMART ORDER: Moved to breakeven for ", orders[i].ticket);
                } else if(orders[i].virtual_sl > open_price && type == POSITION_TYPE_SELL) {
                    orders[i].virtual_sl = open_price - SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 2;
                    Print("SMART ORDER: Moved to breakeven for ", orders[i].ticket);
                }
            }
            
            // Trailing stop logic
            if(orders[i].trailing_step > 0 && profit_pips > orders[i].trailing_step) {
                double new_sl = 0;
                if(type == POSITION_TYPE_BUY) {
                    new_sl = current_price - orders[i].trailing_step;
                    if(new_sl > orders[i].virtual_sl) {
                        orders[i].virtual_sl = new_sl;
                        Print("SMART ORDER: Trailing SL updated for ", orders[i].ticket, " to ", new_sl);
                    }
                } else {
                    new_sl = current_price + orders[i].trailing_step;
                    if(new_sl < orders[i].virtual_sl || orders[i].virtual_sl == 0) {
                        orders[i].virtual_sl = new_sl;
                        Print("SMART ORDER: Trailing SL updated for ", orders[i].ticket, " to ", new_sl);
                    }
                }
            }
            
            // Check virtual SL/TP
            if(type == POSITION_TYPE_BUY) {
                if(orders[i].virtual_sl > 0 && bid <= orders[i].virtual_sl) {
                    trade.PositionClose(orders[i].ticket);
                    Print("SMART ORDER: Virtual SL hit for BUY ", orders[i].ticket);
                    orders[i].active = false;
                }
                if(orders[i].virtual_tp > 0 && bid >= orders[i].virtual_tp) {
                    trade.PositionClose(orders[i].ticket);
                    Print("SMART ORDER: Virtual TP hit for BUY ", orders[i].ticket);
                    orders[i].active = false;
                }
            } else {
                if(orders[i].virtual_sl > 0 && ask >= orders[i].virtual_sl) {
                    trade.PositionClose(orders[i].ticket);
                    Print("SMART ORDER: Virtual SL hit for SELL ", orders[i].ticket);
                    orders[i].active = false;
                }
                if(orders[i].virtual_tp > 0 && ask <= orders[i].virtual_tp) {
                    trade.PositionClose(orders[i].ticket);
                    Print("SMART ORDER: Virtual TP hit for SELL ", orders[i].ticket);
                    orders[i].active = false;
                }
            }
        }
    }
    
    void UpdateVirtualLevels(ulong ticket, double new_sl, double new_tp) {
        for(int i = 0; i < max_orders; i++) {
            if(orders[i].active && orders[i].ticket == ticket) {
                orders[i].virtual_sl = new_sl;
                orders[i].virtual_tp = new_tp;
                return;
            }
        }
    }
};

// ============================================================================
// LOCAL RISK MANAGER
// ============================================================================

class CLocalRiskManager {
private:
    double daily_pnl;
    double max_daily_loss;
    double max_risk_per_trade;
    double current_exposure;
    datetime last_reset_day;
    bool emergency_stop;
    
public:
    CLocalRiskManager(double max_loss, double max_risk) {
        max_daily_loss = max_loss;
        max_risk_per_trade = max_risk;
        daily_pnl = 0;
        current_exposure = 0;
        emergency_stop = false;
        last_reset_day = TimeCurrent();
    }
    
    void Update() {
        // Reset daily PnL at new day
        MqlDateTime dt;
        TimeToStruct(TimeCurrent(), dt);
        MqlDateTime last_dt;
        TimeToStruct(last_reset_day, last_dt);
        
        if(dt.day != last_dt.day) {
            daily_pnl = 0;
            last_reset_day = TimeCurrent();
            emergency_stop = false;
            Print("RISK: Daily PnL reset");
        }
        
        // Calculate current exposure
        current_exposure = 0;
        for(int i = 0; i < PositionsTotal(); i++) {
            if(PositionSelectByTicket(PositionGetTicket(i))) {
                current_exposure += PositionGetDouble(POSITION_VOLUME);
            }
        }
        
        // Check daily loss
        double current_pnl = AccountInfoDouble(ACCOUNT_PROFIT);
        if(current_pnl < -max_daily_loss) {
            if(!emergency_stop) {
                Print("RISK: Maximum daily loss exceeded! Emergency stop activated.");
                emergency_stop = true;
            }
        }
    }
    
    bool CheckRisk(string action, double size) {
        Update();
        
        if(emergency_stop) {
            Print("RISK: Emergency stop active - no new trades");
            return false;
        }
        
        // Check max positions
        if(PositionsTotal() >= 100) {
            Print("RISK: Maximum positions reached");
            return false;
        }
        
        // Check exposure
        double max_exposure = AccountInfoDouble(ACCOUNT_BALANCE) * max_risk_per_trade / 100.0;
        if(current_exposure + size > max_exposure * 10) {
            Print("RISK: Exposure would exceed limit");
            return false;
        }
        
        return true;
    }
    
    void EmergencyCloseAll() {
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            trade.PositionClose(ticket);
            Print("RISK: Emergency closed position ", ticket);
        }
    }
    
    bool IsEmergencyStop() { return emergency_stop; }
    double GetDailyPnL() { return AccountInfoDouble(ACCOUNT_PROFIT); }
    double GetExposure() { return current_exposure; }
};

// ============================================================================
// SIMPLE NEURAL NETWORK (On-Device ML)
// ============================================================================

class CSimpleNeuralNetwork {
private:
    double weights_ih[];  // Flattened: [input_size * hidden_size]
    double weights_ho[];  // Flattened: [hidden_size * output_size]
    double biases_h[];
    double biases_o[];
    int input_size;
    int hidden_size;
    int output_size;
    
    double Sigmoid(double x) {
        if(x > 20) return 1.0;
        if(x < -20) return 0.0;
        return 1.0 / (1.0 + MathExp(-x));
    }
    
    // Helper to access 2D array stored as 1D
    int GetIdx_IH(int i, int j) { return i * hidden_size + j; }
    int GetIdx_HO(int i, int j) { return i * output_size + j; }
    
public:
    CSimpleNeuralNetwork() {
        input_size = 5;
        hidden_size = 8;
        output_size = 3;
        
        ArrayResize(weights_ih, input_size * hidden_size);
        ArrayResize(weights_ho, hidden_size * output_size);
        ArrayResize(biases_h, hidden_size);
        ArrayResize(biases_o, output_size);
        
        // Initialize with small random weights
        MathSrand(GetTickCount());
        for(int i = 0; i < input_size; i++) {
            for(int j = 0; j < hidden_size; j++) {
                weights_ih[GetIdx_IH(i, j)] = (MathRand() / 32768.0 - 0.5) * 0.2;
            }
        }
        for(int i = 0; i < hidden_size; i++) {
            for(int j = 0; j < output_size; j++) {
                weights_ho[GetIdx_HO(i, j)] = (MathRand() / 32768.0 - 0.5) * 0.2;
            }
            biases_h[i] = 0;
        }
        for(int j = 0; j < output_size; j++) biases_o[j] = 0;
    }
    
    void Predict(double &inputs[], double &outputs[]) {
        ArrayResize(outputs, output_size);
        
        // Hidden layer
        double hidden[];
        ArrayResize(hidden, hidden_size);
        
        for(int j = 0; j < hidden_size; j++) {
            hidden[j] = biases_h[j];
            for(int i = 0; i < input_size && i < ArraySize(inputs); i++) {
                hidden[j] += inputs[i] * weights_ih[GetIdx_IH(i, j)];
            }
            hidden[j] = Sigmoid(hidden[j]);
        }
        
        // Output layer
        for(int k = 0; k < output_size; k++) {
            outputs[k] = biases_o[k];
            for(int j = 0; j < hidden_size; j++) {
                outputs[k] += hidden[j] * weights_ho[GetIdx_HO(j, k)];
            }
            outputs[k] = Sigmoid(outputs[k]);
        }
    }
    
    int PredictAction(double &inputs[]) {
        double outputs[];
        Predict(inputs, outputs);
        
        int best_action = 0;
        double best_value = outputs[0];
        for(int i = 1; i < output_size; i++) {
            if(outputs[i] > best_value) {
                best_value = outputs[i];
                best_action = i;
            }
        }
        
        return best_action; // 0=HOLD, 1=BUY, 2=SELL
    }
};

// ============================================================================
// PATTERN RECOGNIZER
// ============================================================================

class CPatternRecognizer {
private:
    double pattern_db[][50]; // Store patterns
    int pattern_labels[];
    int num_patterns;
    int max_patterns;
    
public:
    CPatternRecognizer() {
        max_patterns = 100;
        num_patterns = 0;
        ArrayResize(pattern_db, max_patterns);
        ArrayResize(pattern_labels, max_patterns);
    }
    
    void LearnPattern(double &prices[], int pattern_type) {
        if(num_patterns >= max_patterns) return;
        
        int len = MathMin(50, ArraySize(prices));
        for(int i = 0; i < len; i++) {
            pattern_db[num_patterns][i] = prices[i];
        }
        pattern_labels[num_patterns] = pattern_type;
        num_patterns++;
    }
    
    double CalculateSimilarity(double &pattern1[], double &pattern2[], int len) {
        double dot = 0, norm1 = 0, norm2 = 0;
        
        for(int i = 0; i < len; i++) {
            dot += pattern1[i] * pattern2[i];
            norm1 += pattern1[i] * pattern1[i];
            norm2 += pattern2[i] * pattern2[i];
        }
        
        if(norm1 == 0 || norm2 == 0) return 0;
        return dot / (MathSqrt(norm1) * MathSqrt(norm2));
    }
    
    int RecognizePattern(double &prices[]) {
        if(num_patterns == 0) return -1;
        
        double best_sim = -1;
        int best_label = -1;
        int len = MathMin(50, ArraySize(prices));
        
        for(int p = 0; p < num_patterns; p++) {
            double stored[];
            ArrayResize(stored, len);
            for(int i = 0; i < len; i++) stored[i] = pattern_db[p][i];
            
            double sim = CalculateSimilarity(prices, stored, len);
            if(sim > best_sim) {
                best_sim = sim;
                best_label = pattern_labels[p];
            }
        }
        
        if(best_sim > 0.8) return best_label;
        return -1;
    }
};

// ============================================================================
// ANOMALY DETECTOR
// ============================================================================

class CAnomalyDetector {
private:
    double price_history[];
    double volume_history[];
    int history_size;
    int current_idx;
    double price_mean;
    double price_std;
    
public:
    CAnomalyDetector() {
        history_size = 200;
        current_idx = 0;
        price_mean = 0;
        price_std = 0;
        ArrayResize(price_history, history_size);
        ArrayResize(volume_history, history_size);
    }
    
    void Update(double price, double volume) {
        price_history[current_idx] = price;
        volume_history[current_idx] = volume;
        current_idx = (current_idx + 1) % history_size;
        
        // Update statistics
        double sum = 0;
        for(int i = 0; i < history_size; i++) sum += price_history[i];
        price_mean = sum / history_size;
        
        double var = 0;
        for(int i = 0; i < history_size; i++) {
            double diff = price_history[i] - price_mean;
            var += diff * diff;
        }
        price_std = MathSqrt(var / history_size);
    }
    
    bool DetectAnomaly(double current_value) {
        if(price_std <= 0) return false;
        
        double z_score = MathAbs(current_value - price_mean) / price_std;
        return z_score > 3.0; // 3 sigma rule
    }
    
    double CalculateAnomalyScore(double value) {
        if(price_std <= 0) return 0;
        return MathAbs(value - price_mean) / price_std;
    }
};

// ============================================================================
// ADVANCED COMMUNICATION
// ============================================================================

class CAdvancedComm {
private:
    int priority_levels[];
    string message_buffer[];
    int buffer_size;
    int buffer_count;
    
public:
    CAdvancedComm() {
        buffer_size = 100;
        buffer_count = 0;
        ArrayResize(message_buffer, buffer_size);
        ArrayResize(priority_levels, buffer_size);
    }
    
    void QueueMessage(string message, int priority) {
        if(buffer_count >= buffer_size) {
            // Remove lowest priority
            int min_idx = 0;
            for(int i = 1; i < buffer_count; i++) {
                if(priority_levels[i] < priority_levels[min_idx]) min_idx = i;
            }
            // Shift
            for(int i = min_idx; i < buffer_count - 1; i++) {
                message_buffer[i] = message_buffer[i+1];
                priority_levels[i] = priority_levels[i+1];
            }
            buffer_count--;
        }
        
        message_buffer[buffer_count] = message;
        priority_levels[buffer_count] = priority;
        buffer_count++;
        
        // Sort by priority
        for(int i = 0; i < buffer_count - 1; i++) {
            for(int j = i + 1; j < buffer_count; j++) {
                if(priority_levels[j] > priority_levels[i]) {
                    int tmp_p = priority_levels[i];
                    priority_levels[i] = priority_levels[j];
                    priority_levels[j] = tmp_p;
                    
                    string tmp_m = message_buffer[i];
                    message_buffer[i] = message_buffer[j];
                    message_buffer[j] = tmp_m;
                }
            }
        }
    }
    
    string GetNextMessage() {
        if(buffer_count == 0) return "";
        
        string msg = message_buffer[0];
        for(int i = 0; i < buffer_count - 1; i++) {
            message_buffer[i] = message_buffer[i+1];
            priority_levels[i] = priority_levels[i+1];
        }
        buffer_count--;
        
        return msg;
    }
    
    int GetBufferCount() { return buffer_count; }
};

// ============================================================================
// REFLEX ENGINE (The Spine)
// ============================================================================

class CReflexEngine {
private:
    double avg_spread;
    double spread_buffer[20];
    int buf_idx;
    
public:
    CReflexEngine() { buf_idx = 0; avg_spread = 0; }
    
    void Update(double current_spread) {
        spread_buffer[buf_idx] = current_spread;
        buf_idx = (buf_idx + 1) % 20;
        
        double sum = 0;
        for(int i = 0; i < 20; i++) sum += spread_buffer[i];
        avg_spread = sum / 20.0;
    }
    
    bool IsSafe() {
        double current_spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID));
        if(avg_spread > 0 && current_spread > avg_spread * 3.0) {
            Print("REFLEX GUARD: Spread Spike Detected! ", current_spread, " vs ", avg_spread);
            return false;
        }
        return true;
    }
};

// ============================================================================
// HOLOGRAPHIC HUD
// ============================================================================

class CHolographicHUD {
public:
    void DrawZone(string name, double price_start, double price_end, color clr, datetime t1, datetime t2) {
        if(ObjectFind(0, name) < 0) {
            ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, price_start, t2, price_end);
        } else {
            ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
            ObjectSetDouble(0, name, OBJPROP_PRICE, 0, price_start);
            ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
            ObjectSetDouble(0, name, OBJPROP_PRICE, 1, price_end);
        }
        ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
        ObjectSetInteger(0, name, OBJPROP_FILL, true);
        ObjectSetInteger(0, name, OBJPROP_BACK, true);
    }
    
    void DrawLine(string name, double p1, double p2, color clr, datetime t1, datetime t2) {
        if(ObjectFind(0, name) < 0) {
            ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
        } else {
            ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
            ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);
            ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
            ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);
        }
        ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
        ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
        ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
    }
    
    void DrawText(string name, double price, string text, color clr, datetime t) {
        if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
        ObjectSetString(0, name, OBJPROP_TEXT, text);
        ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
        ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 10);
        ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT);
    }
};

// ============================================================================
// INSTANCES
// ============================================================================

CReflexEngine reflex;
CSmartOrderManager smartOrders;
CHolographicHUD hud;
CLocalIntelligence localIntel;
CAdaptiveExecutor adaptiveExec;
CLocalRiskManager *riskManager;
CSimpleNeuralNetwork neuralNet;
CPatternRecognizer patternRec;
CAnomalyDetector anomalyDet;
CAdvancedComm advComm;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Atl4sBridge v3.0 (AGI Ultra-Intelligent) Starting...");
    trade.SetExpertMagicNumber(MagicNumber);
    
    riskManager = new CLocalRiskManager(MaxDailyLoss, MaxRiskPerTrade);
    
    socket_handle = SocketCreate();
    if(socket_handle == INVALID_HANDLE) {
        Print("Error creating socket: ", GetLastError());
        return(INIT_FAILED);
    }
    
    if(!SocketConnect(socket_handle, host, InpPort, 1000)) {
        Print("Error connecting to Python (Port ", InpPort, "): ", GetLastError());
        return(INIT_FAILED);
    }
    
    Print("Connected to Python AGI Brain.");
    Print("Local Intelligence: ", (EnableLocalML ? "ENABLED" : "DISABLED"));
    Print("Adaptive Execution: ", (EnableAdaptiveExecution ? "ENABLED" : "DISABLED"));
    
    EventSetMillisecondTimer(50);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    EventKillTimer();
    SocketClose(socket_handle);
    delete riskManager;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    if(socket_handle == INVALID_HANDLE) return;
    
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Update all systems
    reflex.Update(ask - bid);
    localIntel.UpdateMarketState();
    adaptiveExec.Update();
    riskManager.Update();
    anomalyDet.Update(bid, (double)SymbolInfoInteger(_Symbol, SYMBOL_VOLUME));
    
    // Manage smart orders
    smartOrders.ManageOrders();
    
    // Scan for best profit
    MqlTick tick;
    SymbolInfoTick(_Symbol, tick);
    
    double best_profit = -999999.0;
    long best_ticket = 0;
    
    for(int i = 0; i < PositionsTotal(); i++) {
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            double prof = PositionGetDouble(POSITION_PROFIT);
            if(prof > best_profit) {
                best_profit = prof;
                best_ticket = PositionGetInteger(POSITION_TICKET);
            }
        }
    }
    
    // Enhanced tick message with local intelligence data
    string msg = StringFormat("TICK|%s|%I64d|%.5f|%.5f|%I64d|%.2f|%d|%.2f|%.2f|%I64d|%.2f|%.2f|%.2f|%d\n",
                              _Symbol, tick.time_msc, tick.bid, tick.ask, tick.volume,
                              AccountInfoDouble(ACCOUNT_EQUITY), PositionsTotal(), 
                              AccountInfoDouble(ACCOUNT_PROFIT),
                              best_profit, best_ticket,
                              localIntel.GetConfidence(),
                              localIntel.GetRSI(),
                              riskManager.GetExposure(),
                              (riskManager.IsEmergencyStop() ? 1 : 0));
    
    uchar req[];
    StringToCharArray(msg, req);
    int len = StringLen(msg);
    SocketSend(socket_handle, req, len);
    
    // Send TRADES_JSON for Python VSL/VTP checks
    if(PositionsTotal() > 0) {
        string trades_json = "TRADES_JSON|[";
        bool first = true;
        
        for(int i = 0; i < PositionsTotal(); i++) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetString(POSITION_SYMBOL);
                int type = (int)PositionGetInteger(POSITION_TYPE);
                double profit = PositionGetDouble(POSITION_PROFIT);
                double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
                double sl = PositionGetDouble(POSITION_SL);
                double tp = PositionGetDouble(POSITION_TP);
                double volume = PositionGetDouble(POSITION_VOLUME);
                
                if(!first) trades_json += ",";
                first = false;
                
                trades_json += StringFormat("{\"ticket\":%I64u,\"symbol\":\"%s\",\"type\":%d,\"profit\":%.2f,\"open_price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"volume\":%.2f}",
                                            ticket, sym, type, profit, open_price, sl, tp, volume);
            }
        }
        trades_json += "]\n";
        
        uchar trades_req[];
        StringToCharArray(trades_json, trades_req);
        int trades_len = StringLen(trades_json);
        SocketSend(socket_handle, trades_req, trades_len);
    }
    
    ReadCommands();
}

void OnTimer() {
    ReadCommands();
    
    // Send Heartbeat every 1s (approx every 20 calls at 50ms)
    static int beat_counter = 0;
    beat_counter++;
    if(beat_counter >= 20) {
        if(socket_handle != INVALID_HANDLE) {
            string msg = StringFormat("HEARTBEAT|%s|%I64d\n", _Symbol, GetTickCount64());
            uchar req[];
            StringToCharArray(msg, req);
            SocketSend(socket_handle, req, StringLen(msg));
        }
        beat_counter = 0;
    }
}

//+------------------------------------------------------------------+
//| Command Processor                                                |
//+------------------------------------------------------------------+
void ReadCommands() {
    if(socket_handle == INVALID_HANDLE) return;
    
    uint readable = SocketIsReadable(socket_handle);
    if(readable > 0) {
        uchar rsp[];
        ArrayResize(rsp, readable);
        int len = SocketRead(socket_handle, rsp, readable, 100);
        if(len > 0) {
            string data = CharArrayToString(rsp, 0, len);
            string cmds[];
            StringSplit(data, '\n', cmds);
            
            for(int i = 0; i < ArraySize(cmds); i++) {
                if(StringLen(cmds[i]) > 0) ProcessCommand(cmds[i]);
            }
        }
    }
}

void ProcessCommand(string json) {
    string parts[];
    StringSplit(json, '|', parts);
    if(ArraySize(parts) < 2) return;
    
    string action = parts[0];
    Print("PYTHON CMD: ", action, " | Parts: ", ArraySize(parts));  // DEBUG
    
    // INTELLIGENT EXECUTION
    if(action == "OPEN_TRADE" && reflex.IsSafe() && !riskManager.IsEmergencyStop()) {
        string symbol = parts[1];
        int cmd = (int)StringToInteger(parts[2]);
        double vol = StringToDouble(parts[3]);
        double sl = StringToDouble(parts[4]);
        double tp = StringToDouble(parts[5]);
        
        // Read confidence from Python (6th parameter) if available
        double py_confidence = 0.0;
        if(ArraySize(parts) >= 7) {
            py_confidence = StringToDouble(parts[6]);
            Print("PYTHON CONF: Received confidence = ", py_confidence, " from parts[6]='", parts[6], "' (ArraySize=", ArraySize(parts), ")");
        } else {
            Print("PYTHON CONF: No confidence in message (ArraySize=", ArraySize(parts), ", expected >= 7)");
        }
        
        // Check with risk manager
        if(!riskManager.CheckRisk(action, vol)) {
            Print("TRADE BLOCKED by Risk Manager");
            return;
        }
        
        // Check with adaptive executor if enabled - USE PYTHON CONFIDENCE
        if(EnableAdaptiveExecution) {
            // Use Python confidence if available (>0), otherwise fallback to local
            double conf_to_use = (py_confidence > 0) ? py_confidence : localIntel.GetConfidence();
            if(!adaptiveExec.ShouldExecute((cmd == 0 ? "BUY" : "SELL"), conf_to_use)) {
                Print("TRADE BLOCKED by Adaptive Executor (Conf: ", conf_to_use, ")");
                return;
            }
        }
        
        // Check for anomalies
        double current_price = (cmd == 0) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
        if(anomalyDet.DetectAnomaly(current_price)) {
            Print("TRADE BLOCKED: Anomaly detected in price!");
            return;
        }
        
        bool success = false;
        if(cmd == 0) {
            success = trade.Buy(vol, symbol, 0, sl, tp);
        } else {
            success = trade.Sell(vol, symbol, 0, sl, tp);
        }
        
        if(success) {
            ulong ticket = trade.ResultOrder();
            if(ticket > 0) {
                smartOrders.AddOrder(ticket, sl, tp, 50 * SymbolInfoDouble(symbol, SYMBOL_POINT), 
                                     30 * SymbolInfoDouble(symbol, SYMBOL_POINT));
                Print("Intelligent Order Executed: ", ticket);
            }
        }
    }
    // CLOSE_TRADE - Close a specific trade by ticket
    else if(action == "CLOSE_TRADE") {
        if(ArraySize(parts) >= 2) {
            ulong ticket = (ulong)StringToInteger(parts[1]);
            Print("PYTHON VSL/VTP: Closing Ticket ", ticket);
            
            if(PositionSelectByTicket(ticket)) {
                bool result = trade.PositionClose(ticket);
                if(result) {
                    Print("SUCCESS: Closed ticket ", ticket);
                } else {
                    Print("FAILED: Could not close ticket ", ticket, " Error: ", GetLastError());
                }
            } else {
                Print("WARNING: Ticket ", ticket, " not found or already closed");
            }
        }
    }
    // CLOSE_ALL - Close all positions for a symbol
    else if(action == "CLOSE_ALL") {
        string target_sym = parts[1];
        Print("EMERGENCY: Closing ALL positions for ", target_sym);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetString(POSITION_SYMBOL);
                if(sym == target_sym || target_sym == "ALL") {
                    trade.PositionClose(ticket);
                    Print("Closed position: ", ticket);
                }
            }
        }
    }
    else if(action == "PRUNE_LOSERS") {
        string target_sym = parts[1];
        Print("PRUNING LOSERS for ", target_sym);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetSymbol(i);
                double profit = PositionGetDouble(POSITION_PROFIT);
                
                if((target_sym == "ALL" || sym == target_sym) && profit < -0.50) {
                    trade.PositionClose(ticket);
                    Print("Pruned Losing Ticket ", ticket, " ($", profit, ")");
                }
            }
        }
    }
    else if(action == "HARVEST_WINNERS") {
        string target_sym = parts[1];
        Print("HARVESTING WINNERS for ", target_sym);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetSymbol(i);
                double profit = PositionGetDouble(POSITION_PROFIT);
                
                if((target_sym == "ALL" || sym == target_sym) && profit > 0.50) {
                    trade.PositionClose(ticket);
                    Print("Harvested Winning Ticket ", ticket, " ($", profit, ")");
                }
            }
        }
    }
    else if(action == "CLOSE_ALL") {
        string target_sym = parts[1];
        Print("CLOSING ALL TRADES for ", target_sym);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetSymbol(i);
                if(target_sym == "ALL" || sym == target_sym) {
                    trade.PositionClose(ticket);
                }
            }
        }
    }
    else if(action == "CLOSE_BUYS") {
        string symbol = ArraySize(parts) > 1 ? parts[1] : _Symbol;
        StringToUpper(symbol);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            StringToUpper(pos_sym);
            
            if(pos_sym == symbol && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
                trade.PositionClose(ticket);
                Print("CLOSE_BUYS: Exiting BUY ", ticket);
            }
        }
    }
    else if(action == "CLOSE_SELLS") {
        string symbol = ArraySize(parts) > 1 ? parts[1] : _Symbol;
        StringToUpper(symbol);
        
        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            ulong ticket = PositionGetTicket(i);
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            StringToUpper(pos_sym);
            
            if(pos_sym == symbol && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
                trade.PositionClose(ticket);
                Print("CLOSE_SELLS: Exiting SELL ", ticket);
            }
        }
    }
    else if(action == "EMERGENCY_STOP") {
        Print("EMERGENCY STOP TRIGGERED");
        riskManager.EmergencyCloseAll();
    }
    else if(action == "DRAW_RECT" && ArraySize(parts) >= 8) {
        // Format: DRAW_RECT|symbol|name|p1|p2|t1|t2|color
        // parts[1]=symbol (for routing), parts[2]=name, parts[3]=p1, parts[4]=p2, parts[5]=t1, parts[6]=t2, parts[7]=color
        Print("DRAW_RECT: Drawing ", parts[2], " from ", parts[3], " to ", parts[4]);
        hud.DrawZone(parts[2], StringToDouble(parts[3]), StringToDouble(parts[4]), 
                     (color)StringToInteger(parts[7]), (datetime)StringToInteger(parts[5]), 
                     (datetime)StringToInteger(parts[6]));
    }
    else if(action == "DRAW_LINE" && ArraySize(parts) >= 8) {
        // Format: DRAW_LINE|symbol|name|p1|p2|t1|t2|color
        hud.DrawLine(parts[2], StringToDouble(parts[3]), StringToDouble(parts[4]), 
                     (color)StringToInteger(parts[7]), (datetime)StringToInteger(parts[5]), 
                     (datetime)StringToInteger(parts[6]));
    }
    else if(action == "DRAW_TEXT" && ArraySize(parts) >= 7) {
        // Format: DRAW_TEXT|symbol|name|price|time|text|color
        hud.DrawText(parts[2], StringToDouble(parts[3]), parts[5], 
                     (color)StringToInteger(parts[6]), (datetime)StringToInteger(parts[4]));
    }
}
