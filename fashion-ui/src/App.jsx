import React, { useState, useEffect } from 'react';
import { User, Sparkles, Heart, ThumbsDown, ShoppingBag, CheckCircle, AlertCircle } from 'lucide-react';

const API_BASE = 'http://localhost:8002';

const OutfitRecommendationUI = () => {
  const [userId, setUserId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [gender, setGender] = useState('All');
  const [age, setAge] = useState('');
  const [personalColor, setPersonalColor] = useState('');
  const [season, setSeason] = useState('');

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [toast, setToast] = useState('');
  
  const [serverStatus, setServerStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(true);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 3000);
  };

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/status`);
        if (response.ok) {
          const data = await response.json();
          setServerStatus(data);
        }
      } catch (err) {
        console.error('서버 상태 확인 실패:', err);
      } finally {
        setStatusLoading(false);
      }
    };

    checkServerStatus();
    const interval = setInterval(checkServerStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getRecommendation = async () => {
    if (!prompt.trim()) {
      showToast('스타일을 입력해주세요');
      return;
    }

    setLoading(true);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          gender,
          age: age || undefined,
          personalColor: personalColor || undefined,
          season: season || undefined,
          user_id: userId || undefined
        })
      });

      if (!response.ok) {
        throw new Error('추천 요청 실패');
      }

      const data = await response.json();
      setResults(data);
      showToast(`✨ ${data.mode === 'mcp' ? 'MCP' : 'Direct'} 모드로 추천 완료!`);
    } catch (err) {
      console.error(err);
      showToast('❌ 추천 요청 실패');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-10" style={{
      background: 'linear-gradient(135deg, #faf5ff 0%, #f3e8ff 50%, #e9d5ff 100%)',
      backgroundAttachment: 'fixed',
      fontFamily: "'Pretendard', -apple-system, sans-serif"
    }}>
      
      <div className="max-w-full mx-auto px-4">
        
        {/* Header - 여백 넉넉하게 */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-6xl md:text-7xl font-bold mb-6" style={{
            background: 'linear-gradient(135deg, #9333ea 0%, #c084fc 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontFamily: "'Orbit', 'Pretendard', sans-serif",
            letterSpacing: '-0.03em',
            lineHeight: '1.3'
          }}>
            ✨ AI 패션 스타일리스트
          </h1>
          <p className="text-purple-700 text-xl font-semibold mt-4" style={{
            letterSpacing: '-0.02em',
            lineHeight: '1.8'
          }}>
            당신만을 위한 특별한 스타일 추천
          </p>
        </div>

        {/* Server Status Banner - 크고 여백 넉넉하게 */}
        {!statusLoading && serverStatus && (
          <div className={`mb-16 p-8 rounded-3xl backdrop-blur-md border-3 transition-all duration-300 max-w-5xl mx-auto ${
            serverStatus.mcp_connected 
              ? 'bg-white/90 border-green-300 shadow-xl' 
              : 'bg-white/90 border-yellow-300 shadow-xl'
          }`}>
            <div className="flex flex-col items-center gap-6">
              <div className="flex items-center gap-4">
                {serverStatus.mcp_connected ? (
                  <>
                    <CheckCircle className="w-9 h-9 text-green-600 animate-pulse" />
                    <div className="text-center">
                      <div className="font-bold text-green-900 text-2xl mb-2" style={{ letterSpacing: '-0.02em' }}>
                        🚀 MCP 서버 연결됨
                      </div>
                      <div className="text-base text-green-700 font-semibold">
                        {serverStatus.mcp_tools?.length || 0}개 도구 사용 가능
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-9 h-9 text-yellow-600 animate-pulse" />
                    <div className="text-center">
                      <div className="font-bold text-yellow-900 text-2xl mb-2" style={{ letterSpacing: '-0.02em' }}>
                        ⚡ Direct 모드로 실행 중
                      </div>
                      <div className="text-base text-yellow-700 font-semibold">
                        {serverStatus.last_error || 'MCP 서버 미연결'}
                      </div>
                    </div>
                  </>
                )}
              </div>
              
              {serverStatus.startup_logs && serverStatus.startup_logs.length > 0 && (
                <details className="text-sm w-full">
                  <summary className="cursor-pointer text-purple-700 hover:text-purple-900 font-bold text-center text-lg">
                    📋 로그 보기
                  </summary>
                  <div className="mt-4 p-5 bg-white/70 rounded-2xl border-2 border-purple-200 max-h-40 overflow-y-auto">
                    {serverStatus.startup_logs.slice(-5).map((log, idx) => (
                      <div key={idx} className={`text-sm mb-3 text-center font-semibold ${
                        log.level === 'error' ? 'text-red-600' :
                        log.level === 'warning' ? 'text-yellow-600' :
                        log.level === 'success' ? 'text-green-600' :
                        'text-purple-600'
                      }`} style={{ lineHeight: '1.6' }}>
                        {log.message}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        )}

        {/* Toast - 크고 둥글게 */}
        {toast && (
          <div className="fixed top-8 left-1/2 transform -translate-x-1/2 z-50 animate-fade-in">
            <div className="px-10 py-5 rounded-2xl shadow-2xl font-bold text-center text-lg" style={{
              background: 'linear-gradient(135deg, #9333ea 0%, #c084fc 100%)',
              color: 'white',
              letterSpacing: '-0.02em'
            }}>
              {toast}
            </div>
          </div>
        )}

        {/* Main Container - 여백 넉넉하게 */}
{/* Main Container - 여백 넉넉하게 */}
<div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
  
  {/* Left Panel - 크고 여백 넉넉하게 */}
  <div className="flex flex-col gap-12">
    
    {/* User Info Card */}
    <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-8 border-3 border-purple-200 hover:shadow-2xl transition-all duration-300">
      <div className="flex items-center justify-center gap-5 mb-10">
        <User className="w-7 h-7 text-purple-600" />
        <h2 className="text-2xl font-bold text-purple-900" style={{ letterSpacing: '-0.02em' }}>
          사용자 정보
        </h2>
      </div>
      
      <input
        type="text"
        value={userId}
        placeholder="사용자 ID (선택사항)"
        onChange={(e) => setUserId(e.target.value)}
        className="w-full px-6 py-4 border-3 border-purple-200 rounded-2xl 
                   focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                   transition-all text-center font-semibold text-lg"
        style={{ letterSpacing: '-0.02em' }}
      />
    </div>

    {/* Style Input Card */}
    <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-8 border-3 border-purple-200 hover:shadow-2xl transition-all duration-300">
      <div className="flex items-center justify-center gap-5 mb-10">
        <Sparkles className="w-7 h-7 text-yellow-500 animate-pulse" />
        <h2 className="text-2xl font-bold text-purple-900" style={{ letterSpacing: '-0.02em' }}>
          스타일 입력
        </h2>
      </div>

      {/* ⬇⬇⬇ 바로 여기: 모든 입력 요소를 "한 부모 - flex flex-col gap"으로 묶음 */}
      <div className="flex flex-col gap-12">

        {/* 프롬프트 */}
        <textarea
          value={prompt}
          placeholder="원하는 스타일을 자유롭게 입력하세요&#10;&#10;예: 캐주얼한 데이트 룩, 비즈니스 미팅용 정장..."
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          className="w-full px-6 py-5 border-3 border-purple-200 rounded-2xl 
                     focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                     resize-none transition-all font-medium text-lg"
          style={{ textAlign: 'left', lineHeight: '1.8', letterSpacing: '-0.02em' }}
        />

        {/* 성별 */}
        <div>
          <label className="block text-center text-lg font-bold text-purple-900 mb-5" style={{ letterSpacing: '-0.02em' }}>
            성별
          </label>
          <select
            value={gender}
            onChange={(e) => setGender(e.target.value)}
            className="w-full px-6 py-4 border-3 border-purple-200 rounded-2xl 
                       focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                       transition-all text-center font-semibold text-lg"
            style={{ letterSpacing: '-0.02em' }}
          >
            <option value="All">전체</option>
            <option value="Men">남성</option>
            <option value="Women">여성</option>
          </select>
        </div>

        {/* 나이 + 계절 */}
        <div className="grid grid-cols-2 gap-8">
          <div>
            <label className="block text-center text-lg font-bold text-purple-900 mb-5" style={{ letterSpacing: '-0.02em' }}>
              나이
            </label>
            <input
              type="text"
              value={age}
              placeholder="예: 25"
              onChange={(e) => setAge(e.target.value)}
              className="w-full px-6 py-4 border-3 border-purple-200 rounded-2xl 
                         focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                         transition-all text-center font-semibold text-lg"
              style={{ letterSpacing: '-0.02em' }}
            />
          </div>

          <div>
            <label className="block text-center text-lg font-bold text-purple-900 mb-5" style={{ letterSpacing: '-0.02em' }}>
              계절
            </label>
            <select
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              className="w-full px-6 py-4 border-3 border-purple-200 rounded-2xl 
                         focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                         transition-all text-center font-semibold text-lg"
              style={{ letterSpacing: '-0.02em' }}
            >
              <option value="">선택</option>
              <option value="Spring">봄</option>
              <option value="Summer">여름</option>
              <option value="Fall">가을</option>
              <option value="Winter">겨울</option>
            </select>
          </div>
        </div>

        {/* 퍼스널 컬러 */}
        <div>
          <label className="block text-center text-lg font-bold text-purple-900 mb-5" style={{ letterSpacing: '-0.02em' }}>
            퍼스널 컬러
          </label>
          <input
            type="text"
            value={personalColor}
            placeholder="예: Warm Spring"
            onChange={(e) => setPersonalColor(e.target.value)}
            className="w-full px-6 py-4 border-3 border-purple-200 rounded-2xl 
                       focus:ring-4 focus:ring-purple-300 focus:border-purple-400 
                       transition-all text-center font-semibold text-lg"
            style={{ letterSpacing: '-0.02em' }}
          />
        </div>

        {/* 버튼 */}
        <button
          onClick={getRecommendation}
          disabled={loading}
          className="w-full py-5 rounded-2xl font-bold text-xl text-white shadow-2xl 
                     hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed 
                     transition-all duration-300 hover:transform hover:-translate-y-1"
          style={{
            background: 'linear-gradient(135deg, #c084fc 0%, #a855f7 100%)',
            letterSpacing: '-0.02em'
          }}
        >
          {loading ? '✨ 추천 중...' : '✨ 추천 받기'}
        </button>

      </div>
      {/* ⬆⬆⬆ gap-12이 정확히 적용됨 */}
    </div>
  </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            {!results && !loading && (
              <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-20 text-center border-3 border-purple-200">
                <Sparkles className="w-24 h-24 text-purple-300 mx-auto mb-8 animate-pulse" />
                <p className="text-purple-700 text-2xl font-bold" style={{ letterSpacing: '-0.02em', lineHeight: '1.6' }}>
                  스타일을 입력하고<br/>추천을 받아보세요! 💜
                </p>
              </div>
            )}

            {loading && (
              <div className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-20 text-center border-3 border-purple-200">
                <div className="animate-spin rounded-full h-24 w-24 border-b-4 border-purple-600 mx-auto mb-8"></div>
                <p className="text-purple-700 text-2xl font-bold" style={{ letterSpacing: '-0.02em' }}>AI가 스타일을 분석 중...</p>
              </div>
            )}

            {results && results.outfits && results.outfits.length > 0 && (
              <div className="flex flex-col gap-12">
                <div className="flex flex-col items-center gap-8 mb-10">
                  <h2 className="text-4xl font-bold text-purple-900 text-center" style={{ letterSpacing: '-0.02em' }}>
                    추천 결과 ({results.outfits.length}개)
                  </h2>
                  <div className={`px-7 py-3 rounded-full text-lg font-bold shadow-xl ${
                    results.mode === 'mcp' 
                      ? 'bg-green-100 text-green-800 border-3 border-green-300' 
                      : 'bg-blue-100 text-blue-800 border-3 border-blue-300'
                  }`} style={{ letterSpacing: '-0.02em' }}>
                    {results.mode === 'mcp' ? '🚀 MCP 모드' : '⚡ Direct 모드'}
                  </div>
                </div>

                {results.outfits.map((outfit, idx) => (
                  <div key={idx} className="bg-white/95 backdrop-blur-md rounded-3xl shadow-2xl p-8 hover:shadow-2xl transition-all duration-300 border-3 border-purple-200">
                    <div className="flex flex-col lg:flex-row items-center gap-12">
                      
                      {/* Outfit Image */}
                      <div className="flex-shrink-0">
                        {outfit.collage ? (
                          <img
                            src={outfit.collage}
                            alt={`Outfit ${idx + 1}`}
                            className="w-96 h-96 object-contain bg-purple-50 rounded-3xl shadow-xl"
                          />
                        ) : (
                          <div className="w-96 h-96 bg-purple-50 rounded-3xl flex items-center justify-center">
                            <p className="text-purple-400 font-bold text-lg">이미지 없음</p>
                          </div>
                        )}
                      </div>

                      {/* Outfit Details */}
                      <div className="flex-1 text-center lg:text-left">
                        <h3 className="text-3xl font-bold text-purple-900 mb-10" style={{ letterSpacing: '-0.02em' }}>
                          코디 #{idx + 1}
                        </h3>

                        {/* Validation Score */}
                        {outfit.scores && (
                          <div className="mb-10 p-6 bg-purple-50 rounded-2xl border-3 border-purple-200">
                            <div className="flex items-center justify-between mb-4">
                              <span className="text-lg font-bold text-purple-800" style={{ letterSpacing: '-0.02em' }}>검증 점수</span>
                              <span className={`text-3xl font-bold ${
                                outfit.scores.accepted ? 'text-green-600' : 'text-yellow-600'
                              }`}>
                                {(outfit.scores.validation * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className={`text-base font-bold px-6 py-3 rounded-xl inline-block ${
                              outfit.scores.accepted 
                                ? 'bg-green-100 text-green-800 border-3 border-green-300' 
                                : 'bg-yellow-100 text-yellow-800 border-3 border-yellow-300'
                            }`} style={{ letterSpacing: '-0.02em' }}>
                              {outfit.scores.accepted ? '✓ 추천 코디' : '⚠ 주의 필요'}
                            </div>
                          </div>
                        )}

                        {/* Items */}
                        <div className="flex flex-col gap-12">
                          {Object.entries(outfit.items || {}).map(([category, item]) => (
                            <div key={category} className="flex flex-col lg:flex-row items-center justify-between text-lg bg-white/70 p-5 rounded-2xl border-2 border-purple-200">
                              <div className="text-center lg:text-left mb-3 lg:mb-0">
                                <span className="font-bold text-purple-900 capitalize" style={{ letterSpacing: '-0.02em' }}>
                                  {category}:
                                </span>
                                <span className="text-purple-700 ml-3 font-semibold" style={{ letterSpacing: '-0.02em' }}>
                                  {item.name}
                                </span>
                              </div>
                              {item.image_url && (
                                <a 
                                  href={item.image_url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-base text-blue-600 hover:text-blue-800 font-bold hover:underline"
                                  style={{ letterSpacing: '-0.02em' }}
                                >
                                  
                                  🔗 Google image에서 검색
                                </a>
                              )}
                            </div>
                          ))}
                        </div>

                        {/* Action Buttons */}
                        <div className="flex flex-wrap justify-center lg:justify-start gap-6">
                          <button
                            onClick={() => showToast('👍 좋아요!')}
                            className="flex items-center gap-3 px-7 py-3.5 bg-pink-100 text-pink-800 rounded-2xl hover:bg-pink-200 transition-all font-bold border-3 border-pink-300 hover:transform hover:-translate-y-1 text-lg"
                            style={{ letterSpacing: '-0.02em' }}
                          >
                            <Heart className="w-5 h-5" />
                            <span>좋아요</span>
                          </button>
                          <button
                            onClick={() => showToast('👎 피드백 저장됨')}
                            className="flex items-center gap-3 px-7 py-3.5 bg-gray-100 text-gray-800 rounded-2xl hover:bg-gray-200 transition-all font-bold border-3 border-gray-300 hover:transform hover:-translate-y-1 text-lg"
                            style={{ letterSpacing: '-0.02em' }}
                          >
                            <ThumbsDown className="w-5 h-5" />
                            <span>싫어요</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OutfitRecommendationUI;