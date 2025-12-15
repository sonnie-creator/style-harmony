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
    <div style={{
      minHeight: '100vh',
      padding: '2.5rem 1.5rem',
      background: 'linear-gradient(135deg, #faf5ff 0%, #f3e8ff 50%, #e9d5ff 100%)',
      backgroundAttachment: 'fixed',
      fontFamily: "'Pretendard', -apple-system, sans-serif"
    }}>
      
      <div style={{ maxWidth: '100%', margin: '0 auto', padding: '0 1rem' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h1 style={{
            fontSize: 'clamp(2.5rem, 5vw, 4rem)',
            fontWeight: 'bold',
            marginBottom: '1.5rem',
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
          <p style={{
            color: '#7e22ce',
            fontSize: '1.25rem',
            fontWeight: '600',
            marginTop: '1rem',
            letterSpacing: '-0.02em',
            lineHeight: '1.8'
          }}>
            당신만을 위한 특별한 스타일 추천
          </p>
        </div>

        {/* Server Status Banner */}
        {!statusLoading && serverStatus && (
          <div style={{
            marginBottom: '4rem',
            padding: '2rem',
            borderRadius: '1.5rem',
            backdropFilter: 'blur(12px)',
            border: serverStatus.mcp_connected ? '3px solid #86efac' : '3px solid #fde047',
            background: serverStatus.mcp_connected ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.9)',
            boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
            maxWidth: '1200px',
            margin: '0 auto 4rem auto'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                {serverStatus.mcp_connected ? (
                  <>
                    <CheckCircle style={{ width: '2.25rem', height: '2.25rem', color: '#16a34a' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontWeight: 'bold', color: '#166534', fontSize: '1.5rem', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
                        🚀 MCP 서버 연결됨
                      </div>
                      <div style={{ fontSize: '1rem', color: '#15803d', fontWeight: '600' }}>
                        {serverStatus.mcp_tools?.length || 0}개 도구 사용 가능
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <AlertCircle style={{ width: '2.25rem', height: '2.25rem', color: '#ca8a04' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontWeight: 'bold', color: '#854d0e', fontSize: '1.5rem', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
                        ⚡ Direct 모드로 실행 중
                      </div>
                      <div style={{ fontSize: '1rem', color: '#a16207', fontWeight: '600' }}>
                        {serverStatus.last_error || 'MCP 서버 미연결'}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Toast */}
        {toast && (
          <div style={{
            position: 'fixed',
            top: '2rem',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 9999,
            padding: '1.25rem 2.5rem',
            borderRadius: '1.25rem',
            boxShadow: '0 10px 35px rgba(147, 51, 234, 0.45)',
            fontWeight: 'bold',
            fontSize: '1.125rem',
            textAlign: 'center',
            background: 'linear-gradient(135deg, #9333ea 0%, #c084fc 100%)',
            color: 'white',
            letterSpacing: '-0.02em'
          }}>
            {toast}
          </div>
        )}

        {/* Main Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '3rem',
          maxWidth: '100%'
        }}>
          
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
            
            {/* User Info Card */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(12px)',
              borderRadius: '1.5rem',
              boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
              padding: '2rem',
              border: '3px solid rgba(192, 132, 252, 0.2)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.25rem', marginBottom: '2.5rem' }}>
                <User style={{ width: '1.75rem', height: '1.75rem', color: '#9333ea' }} />
                <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4c1d95', letterSpacing: '-0.02em', margin: 0 }}>
                  사용자 정보
                </h2>
              </div>
              
              <input
                type="text"
                value={userId}
                placeholder="사용자 ID (선택사항)"
                onChange={(e) => setUserId(e.target.value)}
                style={{
                  width: '100%',
                  padding: '1rem 1.5rem',
                  border: '3px solid #e9d5ff',
                  borderRadius: '1rem',
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  textAlign: 'center',
                  letterSpacing: '-0.02em',
                  transition: 'all 0.3s ease'
                }}
              />
            </div>

            {/* Style Input Card */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(12px)',
              borderRadius: '1.5rem',
              boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
              padding: '2rem',
              border: '3px solid rgba(192, 132, 252, 0.2)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.25rem', marginBottom: '2.5rem' }}>
                <Sparkles style={{ width: '1.75rem', height: '1.75rem', color: '#eab308' }} />
                <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4c1d95', letterSpacing: '-0.02em', margin: 0 }}>
                  스타일 입력
                </h2>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                
                <textarea
                  value={prompt}
                  placeholder="원하는 스타일을 자유롭게 입력하세요&#10;&#10;예: 캐주얼한 데이트 룩, 비즈니스 미팅용 정장..."
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={5}
                  style={{
                    width: '100%',
                    padding: '1.25rem 1.5rem',
                    border: '3px solid #e9d5ff',
                    borderRadius: '1rem',
                    fontSize: '1rem',
                    fontWeight: '500',
                    resize: 'vertical',
                    textAlign: 'left',
                    lineHeight: '1.8',
                    letterSpacing: '-0.02em'
                  }}
                />

                <div>
                  <label style={{ display: 'block', textAlign: 'center', fontSize: '1.125rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>
                    성별
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '1rem 1.5rem',
                      border: '3px solid #e9d5ff',
                      borderRadius: '1rem',
                      fontSize: '1.125rem',
                      fontWeight: '600',
                      textAlign: 'center',
                      letterSpacing: '-0.02em'
                    }}
                  >
                    <option value="All">전체</option>
                    <option value="Men">남성</option>
                    <option value="Women">여성</option>
                  </select>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div>
                    <label style={{ display: 'block', textAlign: 'center', fontSize: '1.125rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>
                      나이
                    </label>
                    <input
                      type="text"
                      value={age}
                      placeholder="예: 25"
                      onChange={(e) => setAge(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '1rem 1.5rem',
                        border: '3px solid #e9d5ff',
                        borderRadius: '1rem',
                        fontSize: '1.125rem',
                        fontWeight: '600',
                        textAlign: 'center',
                        letterSpacing: '-0.02em'
                      }}
                    />
                  </div>

                  <div>
                    <label style={{ display: 'block', textAlign: 'center', fontSize: '1.125rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>
                      계절
                    </label>
                    <select
                      value={season}
                      onChange={(e) => setSeason(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '1rem 1.5rem',
                        border: '3px solid #e9d5ff',
                        borderRadius: '1rem',
                        fontSize: '1.125rem',
                        fontWeight: '600',
                        textAlign: 'center',
                        letterSpacing: '-0.02em'
                      }}
                    >
                      <option value="">선택</option>
                      <option value="Spring">봄</option>
                      <option value="Summer">여름</option>
                      <option value="Fall">가을</option>
                      <option value="Winter">겨울</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', textAlign: 'center', fontSize: '1.125rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>
                    퍼스널 컬러
                  </label>
                  <input
                    type="text"
                    value={personalColor}
                    placeholder="예: Warm Spring"
                    onChange={(e) => setPersonalColor(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '1rem 1.5rem',
                      border: '3px solid #e9d5ff',
                      borderRadius: '1rem',
                      fontSize: '1.125rem',
                      fontWeight: '600',
                      textAlign: 'center',
                      letterSpacing: '-0.02em'
                    }}
                  />
                </div>

                <button
                  onClick={getRecommendation}
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '1.25rem 2rem',
                    borderRadius: '1rem',
                    border: 'none',
                    fontWeight: 'bold',
                    fontSize: '1.25rem',
                    color: 'white',
                    background: 'linear-gradient(135deg, #c084fc 0%, #a855f7 100%)',
                    boxShadow: '0 8px 25px rgba(168, 85, 247, 0.3)',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.6 : 1,
                    letterSpacing: '-0.02em',
                    transition: 'all 0.3s ease'
                  }}
                >
                  {loading ? '✨ 추천 중...' : '✨ 추천 받기'}
                </button>

              </div>
            </div>
          </div>

          {/* Right Column */}
          <div style={{ gridColumn: 'span 2' }}>
            {!results && !loading && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(12px)',
                borderRadius: '1.5rem',
                boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
                padding: '5rem 3rem',
                textAlign: 'center',
                border: '3px solid rgba(192, 132, 252, 0.2)'
              }}>
                <Sparkles style={{ width: '6rem', height: '6rem', color: '#d8b4fe', margin: '0 auto 2rem auto' }} />
                <p style={{ color: '#7e22ce', fontSize: '1.5rem', fontWeight: 'bold', letterSpacing: '-0.02em', lineHeight: '1.6' }}>
                  스타일을 입력하고<br/>추천을 받아보세요! 💜
                </p>
              </div>
            )}

            {loading && (
              <div style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(12px)',
                borderRadius: '1.5rem',
                boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
                padding: '5rem 3rem',
                textAlign: 'center',
                border: '3px solid rgba(192, 132, 252, 0.2)'
              }}>
                <div style={{
                  width: '6rem',
                  height: '6rem',
                  border: '4px solid rgba(168, 85, 247, 0.2)',
                  borderTopColor: '#a855f7',
                  borderRadius: '50%',
                  animation: 'spin 0.8s linear infinite',
                  margin: '0 auto 2rem auto'
                }} />
                <p style={{ color: '#7e22ce', fontSize: '1.5rem', fontWeight: 'bold', letterSpacing: '-0.02em' }}>
                  AI가 스타일을 분석 중...
                </p>
              </div>
            )}

            {results && results.outfits && results.outfits.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                  <h2 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '2rem', letterSpacing: '-0.02em' }}>
                    추천 결과 ({results.outfits.length}개)
                  </h2>
                  <div style={{
                    display: 'inline-block',
                    padding: '0.75rem 1.75rem',
                    borderRadius: '2rem',
                    fontSize: '1.125rem',
                    fontWeight: 'bold',
                    boxShadow: '0 4px 15px rgba(168, 85, 247, 0.2)',
                    background: results.mode === 'mcp' ? '#d1fae5' : '#dbeafe',
                    color: results.mode === 'mcp' ? '#065f46' : '#1e40af',
                    border: results.mode === 'mcp' ? '3px solid #86efac' : '3px solid #93c5fd',
                    letterSpacing: '-0.02em'
                  }}>
                    {results.mode === 'mcp' ? '🚀 MCP 모드' : '⚡ Direct 모드'}
                  </div>
                </div>

                {results.outfits.map((outfit, idx) => (
                  <div key={idx} style={{
                    background: 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: '1.5rem',
                    boxShadow: '0 10px 40px rgba(168, 85, 247, 0.15)',
                    padding: '2.5rem',
                    border: '3px solid rgba(192, 132, 252, 0.2)'
                  }}>
                    <div style={{ display: 'flex', flexDirection: window.innerWidth < 1024 ? 'column' : 'row', alignItems: 'center', gap: '3rem' }}>
                      
                      <div style={{ flexShrink: 0 }}>
                        {outfit.collage ? (
                          <img
                            src={outfit.collage}
                            alt={`Outfit ${idx + 1}`}
                            style={{
                              width: '24rem',
                              height: '24rem',
                              objectFit: 'contain',
                              background: '#faf5ff',
                              borderRadius: '1.5rem',
                              boxShadow: '0 6px 20px rgba(168, 85, 247, 0.12)'
                            }}
                          />
                        ) : (
                          <div style={{
                            width: '24rem',
                            height: '24rem',
                            background: '#faf5ff',
                            borderRadius: '1.5rem',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}>
                            <p style={{ color: '#d8b4fe', fontWeight: 'bold', fontSize: '1.125rem' }}>이미지 없음</p>
                          </div>
                        )}
                      </div>

                      <div style={{ flex: 1 }}>
                        <h3 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4c1d95', marginBottom: '2.5rem', letterSpacing: '-0.02em' }}>
                          코디 #{idx + 1}
                        </h3>

                        {outfit.scores && (
                          <div style={{
                            marginBottom: '2.5rem',
                            padding: '1.5rem',
                            background: '#faf5ff',
                            borderRadius: '1rem',
                            border: '3px solid #e9d5ff'
                          }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                              <span style={{ fontSize: '1.125rem', fontWeight: 'bold', color: '#7e22ce', letterSpacing: '-0.02em' }}>검증 점수</span>
                              <span style={{
                                fontSize: '2rem',
                                fontWeight: 'bold',
                                color: outfit.scores.accepted ? '#16a34a' : '#ca8a04'
                              }}>
                                {(outfit.scores.validation * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div style={{
                              display: 'inline-block',
                              fontSize: '1rem',
                              fontWeight: 'bold',
                              padding: '0.75rem 1.5rem',
                              borderRadius: '0.75rem',
                              background: outfit.scores.accepted ? '#d1fae5' : '#fef3c7',
                              color: outfit.scores.accepted ? '#065f46' : '#854d0e',
                              border: outfit.scores.accepted ? '3px solid #86efac' : '3px solid #fde047',
                              letterSpacing: '-0.02em'
                            }}>
                              {outfit.scores.accepted ? '✓ 추천 코디' : '⚠ 주의 필요'}
                            </div>
                          </div>
                        )}

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginBottom: '2.5rem' }}>
                          {Object.entries(outfit.items || {}).map(([category, item]) => (
                            <div key={category} style={{
                              display: 'flex',
                              flexDirection: window.innerWidth < 768 ? 'column' : 'row',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              fontSize: '1.125rem',
                              background: 'rgba(255, 255, 255, 0.7)',
                              padding: '1.25rem',
                              borderRadius: '1rem',
                              border: '2px solid #e9d5ff',
                              gap: '1rem'
                            }}>
                              <div style={{ textAlign: window.innerWidth < 768 ? 'center' : 'left' }}>
                                <span style={{ fontWeight: 'bold', color: '#4c1d95', letterSpacing: '-0.02em', textTransform: 'capitalize' }}>
                                  {category}:
                                </span>
                                <span style={{ color: '#7e22ce', marginLeft: '0.75rem', fontWeight: '600', letterSpacing: '-0.02em' }}>
                                  {item.name}
                                </span>
                              </div>
                              {item.hm_url && (
                                <a 
                                  href={item.hm_url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  style={{
                                    fontSize: '1rem',
                                    color: '#2563eb',
                                    fontWeight: 'bold',
                                    textDecoration: 'none',
                                    letterSpacing: '-0.02em'
                                  }}
                                >
                                  🔗 Google에서 검색
                                </a>
                              )}
                            </div>
                          ))}
                        </div>

                        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: window.innerWidth < 768 ? 'center' : 'flex-start', gap: '1.5rem' }}>
                          <button
                            onClick={() => showToast('👍 좋아요!')}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.75rem',
                              padding: '1rem 1.75rem',
                              background: '#fce7f3',
                              color: '#9f1239',
                              borderRadius: '1rem',
                              border: '3px solid #fbcfe8',
                              fontWeight: 'bold',
                              fontSize: '1.125rem',
                              cursor: 'pointer',
                              letterSpacing: '-0.02em',
                              transition: 'all 0.3s ease'
                            }}
                          >
                            <Heart style={{ width: '1.25rem', height: '1.25rem' }} />
                            <span>좋아요</span>
                          </button>
                          <button
                            onClick={() => showToast('👎 피드백 저장됨')}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.75rem',
                              padding: '1rem 1.75rem',
                              background: '#f3f4f6',
                              color: '#374151',
                              borderRadius: '1rem',
                              border: '3px solid #d1d5db',
                              fontWeight: 'bold',
                              fontSize: '1.125rem',
                              cursor: 'pointer',
                              letterSpacing: '-0.02em',
                              transition: 'all 0.3s ease'
                            }}
                          >
                            <ThumbsDown style={{ width: '1.25rem', height: '1.25rem' }} />
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
      
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default OutfitRecommendationUI;