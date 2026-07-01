import { AlphaZeroAgent } from '../src/agent.js';
import http from 'node:http';
import { EventEmitter } from 'node:events';
import {
  jest,
  describe,
  it,
  beforeEach,
  afterEach,
  expect,
} from '@jest/globals';

describe('AlphaZeroAgent', () => {
  let agent: AlphaZeroAgent;

  let originalHttpRequest: unknown;

  beforeEach(() => {
    agent = new AlphaZeroAgent('/tmp/test.sock');
    originalHttpRequest = http.request;
    jest.resetAllMocks();
  });

  afterEach(() => {
    http.request = originalHttpRequest as typeof http.request;
  });

  function mockHttpResponse(responseData: unknown) {
    const mockReq = new EventEmitter() as EventEmitter & {
      write: jest.Mock;
      end: jest.Mock;
      setTimeout: jest.Mock;
      destroy: jest.Mock;
    };
    mockReq.write = jest.fn();
    mockReq.end = jest.fn();
    mockReq.setTimeout = jest.fn();
    mockReq.destroy = jest.fn();

    const mockRes = new EventEmitter();

    http.request = ((options: unknown, callback: unknown) => {
      if (typeof callback === 'function') {
        callback(mockRes);
      }
      return mockReq;
    }) as unknown as typeof http.request;

    return { mockReq, mockRes, responseData };
  }

  it('should parse dense policy arrays correctly', async () => {
    const { mockRes, responseData } = mockHttpResponse({
      policy: [0.1, 0.8, 0.05, 0.05],
      value: 0.5,
    });

    const actPromise = agent.act({ test: true });

    mockRes.emit('data', JSON.stringify(responseData));
    mockRes.emit('end');

    const result = await actPromise;
    expect(result).toBe(1); // Index of highest value (0.8)
  });

  it('should parse sparse policy arrays correctly', async () => {
    const { mockRes, responseData } = mockHttpResponse({
      policy: [
        { index: 5, value: 0.1 },
        { index: 42, value: 0.9 },
        { index: 100, value: 0.05 },
      ],
      value: -0.1,
    });

    const actPromise = agent.act({ test: true });

    mockRes.emit('data', JSON.stringify(responseData));
    mockRes.emit('end');

    const result = await actPromise;
    expect(result).toBe(42); // Index with highest value (0.9)
  });
});
