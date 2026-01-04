
//+------------------------------------------------------------------+
//|                                                  Atl4sBridge.mq5 |
//|                                  Copyright 2025, Atl4s Corp AI   |
//|                                             https://atl4s.ai     |
//+------------------------------------------------------------------+
#property copyright "Atl4s Corp AI"
#property link      "https://atl4s.ai"
#property version   "2.10"
#property strict
#include <Trade\Trade.mqh>

// --- INPUTS ---
input int    InpPort = 5557; // User defined port
input int    MagicNumber = 1337;
input int    MaxSlippage = 5;

// --- GLOBALS ---
int socket_handle = INVALID_HANDLE;
string host = "127.0.0.1";
CTrade trade;

// --- CLASSES (The Mirror Neuron Logic) ---

// 1. REFLEX ENGINE (The Spine)
class CReflexEngine {
private:
   double avg_spread;
   double spread_buffer[20];
   int    buf_idx;
   
public:
   CReflexEngine() { buf_idx = 0; avg_spread = 0; }
   
   void Update(double current_spread) {
      spread_buffer[buf_idx] = current_spread;
      buf_idx = (buf_idx + 1) % 20;
      
      double sum = 0;
      for(int i=0; i<20; i++) sum += spread_buffer[i];
      avg_spread = sum / 20.0;
   }
   
   bool IsSafe() {
      // 1. Spread Check
      double current_spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID));
      if (avg_spread > 0 && current_spread > avg_spread * 3.0) {
         Print("REFLEX GUARD: Spread Spike Detected! ", current_spread, " vs ", avg_spread);
         return false;
      }
      return true;
   }
};

// 2. SHADOW BOOK (The Ninja)
struct ShadowOrder {
   ulong ticket;
   double virtual_sl;
   double virtual_tp;
   bool  active;
};

class CShadowBook {
private:
   ShadowOrder orders[];
   
public:
   void AddOrder(ulong ticket, double sl, double tp) {
      int s = ArraySize(orders);
      ArrayResize(orders, s+1);
      orders[s].ticket = ticket;
      orders[s].virtual_sl = sl;
      orders[s].virtual_tp = tp;
      orders[s].active = true;
   }
   
   void Check() {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      for(int i=0; i<ArraySize(orders); i++) {
         if (!orders[i].active) continue;
         
         if (PositionSelectByTicket(orders[i].ticket)) {
            long type = PositionGetInteger(POSITION_TYPE);
            
            // BUY -> Close at Bid
            if (type == POSITION_TYPE_BUY) {
               if (orders[i].virtual_sl > 0 && bid <= orders[i].virtual_sl) {
                  trade.PositionClose(orders[i].ticket);
                  Print("SHADOW SL: Closing BUY ", orders[i].ticket);
                  orders[i].active = false;
               }
               if (orders[i].virtual_tp > 0 && bid >= orders[i].virtual_tp) {
                  trade.PositionClose(orders[i].ticket);
                  Print("SHADOW TP: Closing BUY ", orders[i].ticket);
                  orders[i].active = false;
               }
            }
            // SELL -> Close at Ask
            else if (type == POSITION_TYPE_SELL) {
               if (orders[i].virtual_sl > 0 && ask >= orders[i].virtual_sl) {
                  trade.PositionClose(orders[i].ticket);
                  Print("SHADOW SL: Closing SELL ", orders[i].ticket);
                  orders[i].active = false;
               }
               if (orders[i].virtual_tp > 0 && ask <= orders[i].virtual_tp) {
                  trade.PositionClose(orders[i].ticket);
                  Print("SHADOW TP: Closing SELL ", orders[i].ticket);
                  orders[i].active = false;
               }
            }
         } else {
             orders[i].active = false; // Closed externally
         }
      }
   }
};

// 3. HOLOGRAPHIC HUD
class CHolographicHUD {
public:
   void DrawZone(string name, double price_start, double price_end, color clr) {
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, TimeCurrent(), price_start, TimeCurrent()+PeriodSeconds()*10, price_end);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, name, OBJPROP_FILL, true); 
   }
};

// --- INSTANCES ---
CReflexEngine reflex;
CShadowBook   shadow;
CHolographicHUD hud;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("Atl4sBridge v2.1 (Native Socket 5557) Starting...");
   trade.SetExpertMagicNumber(MagicNumber);
   
   // Create Socket
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE) {
      Print("Error creating socket: ", GetLastError());
      return(INIT_FAILED);
   }
   
   // Connect
   if(!SocketConnect(socket_handle, host, InpPort, 1000)) {
      Print("Error connecting to Python (Port ", InpPort, "): ", GetLastError());
      return(INIT_FAILED);
   }
   
   Print("Connected to Python Brain.");
   EventSetMillisecondTimer(50); // Fast poll
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   SocketClose(socket_handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(socket_handle == INVALID_HANDLE) return;
   
   // 1. Reflex Update
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   reflex.Update(ask - bid);
   
   // 2. Scan for Best Profit (Surgical TP)
   MqlTick tick;
   SymbolInfoTick(_Symbol, tick);
   
   double best_profit = -999999.0;
   long best_ticket = 0;
   
   for(int i=0; i<PositionsTotal(); i++) {
      if(PositionSelectByTicket(PositionGetTicket(i))) {
         // FIX: Scan ALL symbols to ensure cross-pair winners are detected.
         // if(PositionGetSymbol(i) == _Symbol) { 
            double prof = PositionGetDouble(POSITION_PROFIT);
            if(prof > best_profit) {
               best_profit = prof;
               best_ticket = PositionGetInteger(POSITION_TICKET);
            }
         // }
      }
   }
   
   // Msg: TICK|SYMBOL|TIME|BID|ASK|VOL|EQUITY|POS_COUNT|ACC_PROFIT|BEST_PROFIT|BEST_TICKET\n
   string msg = StringFormat("TICK|%s|%I64d|%.5f|%.5f|%I64d|%.2f|%d|%.2f|%.2f|%I64d\n", 
                             _Symbol, tick.time_msc, tick.bid, tick.ask, tick.volume,
                             AccountInfoDouble(ACCOUNT_EQUITY), PositionsTotal(), AccountInfoDouble(ACCOUNT_PROFIT),
                             best_profit, best_ticket);
   
   uchar req[];
   StringToCharArray(msg, req);
   int len = StringLen(msg);
   SocketSend(socket_handle, req, len);

   // 3. Check Shadow Orders
   shadow.Check();
   
   // 4. Check Commands (Fast read)
   ReadCommands();
   
   // 5. Trailing Stop Guard
   CheckTrailing();
  }

void OnTimer() {
   ReadCommands();
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
           StringSplit(data, '\n', cmds); // Handle multiple commands
           
           for(int i=0; i<ArraySize(cmds); i++) {
               if(StringLen(cmds[i]) > 0) ProcessCommand(cmds[i]);
           }
       }
   }
}

void ProcessCommand(string json) {
    // Format: ACTION|SYMBOL|PARAM1|PARAM2...
    string parts[];
    StringSplit(json, '|', parts);
    if(ArraySize(parts) < 2) return;
    
    string action = parts[0];
    
    // EXECUTION LOGIC
    if (action == "OPEN_TRADE" && reflex.IsSafe()) {
         // Format: OPEN_TRADE|SYMBOL|TYPE(0/1)|LOTS|SL|TP
         string symbol = parts[1];
         int cmd = (int)StringToInteger(parts[2]);
         double vol = StringToDouble(parts[3]);
         double sl = StringToDouble(parts[4]);
         double tp = StringToDouble(parts[5]);
         
         // Hard SL/TP (Visible to Broker and User)
         double broker_sl = sl;
         double broker_tp = tp;
         
         double price = 0;
         
         if (cmd == 0) { // BUY
             if(trade.Buy(vol, symbol, 0, broker_sl, broker_tp)) {
                 price = trade.ResultPrice();
             }
         } else { // SELL
             if(trade.Sell(vol, symbol, 0, broker_sl, broker_tp)) {
                 price = trade.ResultPrice();
             }
         }
         
         ulong ticket = trade.ResultRetcode() == TRADE_RETCODE_DONE ? trade.ResultOrder() : 0;
         
         if (ticket > 0) {
             // Visual Indicator: Arrow
             string arrow_name = "ATL4S_EXE_" + (string)ticket;
             ENUM_OBJECT type_obj = (cmd == 0) ? OBJ_ARROW_BUY : OBJ_ARROW_SELL;
             color clr = (cmd == 0) ? clrLime : clrRed;
             
             ObjectCreate(0, arrow_name, type_obj, 0, TimeCurrent(), price);
             ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, clr);
             ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 2);
             
             Print("Hard Order Executed: ", ticket, " | SL: ", sl, " | TP: ", tp);
         }
    }
    else if(action == "PRUNE_LOSERS") {
        // PRUNE_LOSERS|SYMBOL (or ALL)
        string target_sym = parts[1];
        Print("PRUNING LOSERS for ", target_sym);
        
        for(int i=PositionsTotal()-1; i>=0; i--) {
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
        // HARVEST_WINNERS|SYMBOL (or ALL)
        string target_sym = parts[1];
        Print("HARVESTING WINNERS for ", target_sym);
        
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                string sym = PositionGetSymbol(i);
                double profit = PositionGetDouble(POSITION_PROFIT);
                
                // If Target matches (or ALL) AND Profit is positive (Secure the bag)
                if((target_sym == "ALL" || sym == target_sym) && profit > 0.50) {
                     trade.PositionClose(ticket);
                     Print("Harvested Winning Ticket ", ticket, " ($", profit, ")");
                }
            }
        }
    }
    else if(action == "CLOSE_ALL") {
       // CLOSE_ALL|SYMBOL
       string target_sym = parts[1];
       Print("CLOSING ALL TRADES for ", target_sym);
       
       for(int i=PositionsTotal()-1; i>=0; i--) {
          ulong ticket = PositionGetTicket(i);
          if(PositionSelectByTicket(ticket)) {
             string sym = PositionGetSymbol(i);
             if(target_sym == "ALL" || sym == target_sym) {
                trade.PositionClose(ticket);
             }
          }
       }
    }
    else if (action == "CLOSE_BUYS") {
        string symbol = ArraySize(parts) > 1 ? parts[1] : _Symbol;
        StringToUpper(symbol);
        
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong ticket = PositionGetTicket(i);
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            StringToUpper(pos_sym);
            
            if(pos_sym == symbol && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
                trade.PositionClose(ticket);
                Print("CLOSE_BUYS: Exiting BUY ", ticket);
            }
        }
    }
    else if (action == "CLOSE_SELLS") {
        string symbol = ArraySize(parts) > 1 ? parts[1] : _Symbol;
        StringToUpper(symbol);
        
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong ticket = PositionGetTicket(i);
            string pos_sym = PositionGetString(POSITION_SYMBOL);
            StringToUpper(pos_sym);
            
            if(pos_sym == symbol && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
                trade.PositionClose(ticket);
                Print("CLOSE_SELLS: Exiting SELL ", ticket);
            }
        }
    }
    // DRAWING LOGIC
    else if (action == "DRAW_ZONE") {
        // DRAW_ZONE|NAME|P1|P2|COLOR_INT
        if(ArraySize(parts) >= 5) {
            hud.DrawZone(parts[1], StringToDouble(parts[2]), StringToDouble(parts[3]), (color)StringToInteger(parts[4]));
        }
    }
}

//+------------------------------------------------------------------+
//| Trailing Stop Logic                                              |
//+------------------------------------------------------------------+
void CheckTrailing() {
   for(int i=PositionsTotal()-1; i>=0; i--) {
      ulong ticket = PositionGetTicket(i);
      string symbol = PositionGetString(POSITION_SYMBOL);
      
      if (symbol != _Symbol) continue;
      
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double price_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double price_current = PositionGetDouble(POSITION_PRICE_CURRENT);
      long type = PositionGetInteger(POSITION_TYPE);
      
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      
      // Dynamic Trail: 100 points activation, 50 points step
      double trail_dist = 100 * point; 
      double trail_step = 50 * point;
      
      if (type == POSITION_TYPE_BUY) {
          if (price_current - price_open > trail_dist) {
              double new_sl = price_current - trail_step;
              if (new_sl > sl) {
                  trade.PositionModify(ticket, new_sl, tp);
                  Print("TRAILING STOP: Updated BUY ", ticket, " SL to ", new_sl);
              }
          }
      } else if (type == POSITION_TYPE_SELL) {
          if (price_open - price_current > trail_dist) {
              double new_sl = price_current + trail_step;
              if (sl == 0 || new_sl < sl) {
                  trade.PositionModify(ticket, new_sl, tp);
                  Print("TRAILING STOP: Updated SELL ", ticket, " SL to ", new_sl);
              }
          }
      }
   }
}
