package multeval.metrics;

import jannopts.*;

import java.util.*;

/**
 * Computes a metric score given that metric's sufficient statistics (Yes, we're
 * skipping the hard, expensive part and letting the metric reference
 * implementations take care of that)
 * 
 * @author jhclark
 */
public abstract class Metric<Stats extends SuffStats<Stats>> {

	public abstract Stats stats(String sentence, List<String> refs);

	public abstract double score(Stats suffStats);

	public abstract void configure(Configurator opts) throws ConfigurationException;

	public abstract boolean isBiggerBetter();

	// this should include version!
	public abstract String getMetricDescription();

	// indicates that a single instance can safely be used by a single thread
	// does NOT indicate that a single instance can support multiple threads
	// Update: We removed this since TER is now thread-safe in that a single JVM
	// can have multiple instances of it
	//public abstract boolean isThreadsafe();

	public String[] getSubmetricNames() {
		return new String[0];
	}

	public double[] scoreSubmetrics(Stats suffStats) {
		return new double[0];
	}

	// hack around generics by erasure
	@SuppressWarnings("unchecked")
	public double scoreStats(SuffStats<?> suffStats) {
		return score((Stats) suffStats);
	}

	// hack around generics by erasure
	@SuppressWarnings("unchecked")
	public double[] scoreSubmetricsStats(SuffStats<?> suffStats) {
		return scoreSubmetrics((Stats) suffStats);
	}

	public String scoreSubmetricsString(SuffStats<?> suffStats) {
		StringBuilder builder = new StringBuilder();
		String[] names = getSubmetricNames();
		double[] subs = scoreSubmetricsStats(suffStats);
		for (int i = 0; i < names.length; i++) {
			builder.append(String.format("%s=%.2f", names[i], subs[i]));
			if (i < names.length - 1) {
				builder.append("; ");
			}
		}
		return builder.toString();
	}

	public Metric<?> threadClone() {
		// for metrics like meteor that *can* be thread-safe, we need to make
		// copies
		// for metric like ter, we're just completely hosed
		return this; // works if this metric is threadsafe in general
	}
}
