//+------------------------------------------------------------------+
//|                                           Atl4sBridge_Tester.mq5 |
//|                                   Synchro-Bridge for Backtesting |
//|                          Uses Named Pipes for Blocking Comm      |
//+------------------------------------------------------------------+
#property copyright "Atl4s-Forex"
#property version   "1.00"
#property description "Synchronous Bridge for Strategy Tester using Named Pipes"
#property strict

// --- Imports ---
#import "kernel32.dll"
   int CreateFileW(string lpFileName, uint dwDesiredAccess, uint dwShareMode, uint lpSecurityAttributes, uint dwCreationDisposition, uint dwFlagsAndAttributes, int hTemplateFile);
   bool WriteFile(int hFile, const uchar &lpBuffer[], uint nNumberOfBytesToWrite, uint &lpNumberOfBytesWritten, int lpOverlapped);
   bool ReadFile(int hFile, uchar &lpBuffer[], uint nNumberOfBytesToRead, uint &lpNumberOfBytesRead, int lpOverlapped);
   bool CloseHandle(int hObject);
   bool WaitNamedPipeW(string lpNamedPipeName, uint nTimeOut);
   bool CallNamedPipeW(string lpNamedPipeName, const uchar &lpInBuffer[], uint nInBufferSize, uchar &lpOutBuffer[], uint nOutBufferSize, uint &lpBytesRead, uint nTimeOut);
#import

// --- Constants ---
#define GENERIC_READ            0x80000000
#define GENERIC_WRITE           0x40000000
#define OPEN_EXISTING           3
#define INVALID_HANDLE_VALUE    -1
#define PIPE_ACCESS_DUPLEX      3
#define PIPE_TYPE_MESSAGE       4
#define PIPE_READMODE_MESSAGE   2
#define PIPE_WAIT               0

string PIPE_NAME = "\\\\.\\pipe\\Atl4sPipe";

// --- Globals ---
int pipeHandle = INVALID_HANDLE_VALUE;
ulong last_ticket = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Atl4s Tester Bridge: Initializing...");
   
   // We don't connect here because we use CallNamedPipe for atomic transactions in OnTick
   // Or we can keep a persistent connection if we implement a proper loop.
   // For Tester, CallNamedPipe is safer as it opens/writes/reads/closes in one go, 
   // ensuring we don't desync if Python restarts.
   
   // Check if DLLs are allowed
   if(!MQLInfoInteger(MQL_DLLS_ALLOWED)) {
      Alert("Error: DLL imports must be allowed for Named Pipes!");
      return INIT_FAILED;
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Atl4s Tester Bridge: Shutting down.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Gather Data
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   int positions = PositionsTotal();
   
   // 2. Format Message (CSV format for speed)
   // Type|Symbol|Time|Bid|Ask|Vol|Equity|Balance|Positions
   string msg = StringFormat("TICK|%s|%I64d|%.5f|%.5f|%I64d|%.2f|%.2f|%d", 
                  _Symbol, tick.time_msc, tick.bid, tick.ask, tick.volume, equity, balance, positions);
                  
   // 3. Send & Wait for Response (Blocking)
   string response = SendPipeMessage_Blocking(msg);
   
   // 4. Execute Response
   if(StringLen(response) > 0) {
      ProcessCommand(response);
   }
}

//+------------------------------------------------------------------+
//| Send Message via Named Pipe (Blocking Transaction)               |
//+------------------------------------------------------------------+
string SendPipeMessage_Blocking(string message)
{
   uchar req[];
   StringToCharArray(message, req);
   
   uchar resp[1024]; // 1KB buffer for response
   ArrayInitialize(resp, 0);
   
   uint bytesRead = 0;
   
   // CallNamedPipe connects, writes, reads, and closes. Perfect for atomic tick step.
   bool result = CallNamedPipeW(PIPE_NAME, req, ArraySize(req), resp, ArraySize(resp), bytesRead, 20000); // 20s timeout
   
   if(!result) {
      // If failed, maybe Python is busy or not running. 
      // In Tester, we want to retry or fail hard to avoid skipping ticks.
      // Print("Error calling pipe: ", GetLastError());
      return "";
   }
   
   string respStr = CharArrayToString(resp, 0, bytesRead);
   return respStr;
}

//+------------------------------------------------------------------+
//| Process Command                                                  |
//+------------------------------------------------------------------+
void ProcessCommand(string cmd_str)
{
   if(cmd_str == "NO_OP") return;
   
   string parts[];
   StringSplit(cmd_str, '|', parts);
   
   if(ArraySize(parts) < 1) return;
   
   string action = parts[0];
   
   if(action == "OPEN") {
      // OPEN|Type(0=Buy,1=Sell)|Vol|SL|TP
      if(ArraySize(parts) < 5) return;
      
      int type = (int)StringToInteger(parts[1]);
      double vol = StringToDouble(parts[2]);
      double sl = StringToDouble(parts[3]);
      double tp = StringToDouble(parts[4]);
      
      Trade(type, vol, sl, tp);
   }
   else if(action == "CLOSE") {
      // CLOSE|Ticket
       if(ArraySize(parts) < 2) return;
       ulong ticket = (ulong)StringToInteger(parts[1]);
       CloseTrade(ticket);
   }
}

//+------------------------------------------------------------------+
//| Trade Functions                                                  |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
CTrade trade;

void Trade(int type, double vol, double sl, double tp) {
   if(type == 0) trade.Buy(vol, _Symbol, 0, sl, tp, "Atl4s AGI");
   else trade.Sell(vol, _Symbol, 0, sl, tp, "Atl4s AGI");
}

void CloseTrade(ulong ticket) {
   trade.PositionClose(ticket);
}
