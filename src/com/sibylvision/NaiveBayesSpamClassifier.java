package com.sibylvision;

import java.util.HashMap;
import java.util.Set;

public class NaiveBayesSpamClassifier {
	
	private HashMap<String, Integer[]> featureCategory;
	private HashMap<NaiveBayesSpamClassification, Integer> categoryUsage;
	private static final double ASSUMED_PROBABILITY_WEIGHT = 1.0;
	private static final double ASSUMED_PROBABILITY = 0.5;
	
	public NaiveBayesSpamClassifier() {
		featureCategory = new HashMap<String, Integer[]>();
		categoryUsage = new HashMap<NaiveBayesSpamClassification, Integer>();
	}

	public NaiveBayesSpamClassification classify(String document) {
		Double[] values = new Double[2];
		for (NaiveBayesSpamClassification c : categories()) {
			values[c.arrayLocation()] = probabilityOfCategory(document, c);
		}
		
		if (values[0] * NaiveBayesSpamClassification.GOOD_CLASSIFICATION.classificationThreshold() > values[1]) {
			return NaiveBayesSpamClassification.GOOD_CLASSIFICATION;
		} else if (values[1] * NaiveBayesSpamClassification.BAD_CLASSIFICATION.classificationThreshold() > values[0]) {
			return NaiveBayesSpamClassification.BAD_CLASSIFICATION;
		} else {
			return NaiveBayesSpamClassification.NO_CLASSIFICATION; 
		}
	}
	
	public static void main(String[] args) {
		NaiveBayesSpamClassifier self = new NaiveBayesSpamClassifier();
		
		self.train("the quick brown fox jumps", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		self.train("make quick money at the online casino", NaiveBayesSpamClassification.BAD_CLASSIFICATION);
		self.train("Nobody owns the water.", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		self.train("buy pharmaceuticals now", NaiveBayesSpamClassification.BAD_CLASSIFICATION);
		self.train("the quick rabbit jumps fences", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		self.train("the quick brown fox jumps", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		self.train("make quick money at the online casino", NaiveBayesSpamClassification.BAD_CLASSIFICATION);
		self.train("Nobody owns the water.", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		self.train("buy pharmaceuticals now", NaiveBayesSpamClassification.BAD_CLASSIFICATION);
		self.train("the quick rabbit jumps fences", NaiveBayesSpamClassification.GOOD_CLASSIFICATION);
		
		System.out.println(self.classify("what"));
	}
	
	String[] getFeatures(String text) {
		return text.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");
	}
	
	private int increaseFeatureFrequencyClassificationCount(String word, NaiveBayesSpamClassification classification) {
		if (featureCategory.containsKey(word)) {
			Integer[] array = featureCategory.get(word);
			array[classification.arrayLocation()] += 1;
			featureCategory.put(word, array);
			return featureCategory.get(word)[classification.arrayLocation()];
		} else {
			if (classification.arrayLocation() == 0) {
				Integer[] array = new Integer[2];
				array[0] = 1;
				array[1] = 0;
				featureCategory.put(word, array);
				return array[0];
			} else if (classification.arrayLocation() == 1) {
				Integer[] array = new Integer[2];
				array[0] = 0;
				array[1] = 1;
				featureCategory.put(word, array);
				return array[1];
			} else {
				new IllegalArgumentException("Invalid classification enum.");
			}
		}
		return 0;
	}
	
	private int increaseCategoryCount(NaiveBayesSpamClassification classification) {
		if (categoryUsage.containsKey(classification)) {
			categoryUsage.put(classification, categoryUsage.get(classification) + 1);
			return categoryUsage.get(classification);
		} else {
			categoryUsage.put(classification, 1);
			return 1;
		}
	}
	
	private int featureClassificationCount(String word, NaiveBayesSpamClassification classification) {
		if (featureCategory.containsKey(word)) {
			return featureCategory.get(word)[classification.arrayLocation()];
		} else {
			return 0;
		}
	}
	
	private double classificationCount(NaiveBayesSpamClassification classification) {
		if (categoryUsage.containsKey(classification)) {
			return categoryUsage.get(classification);
		} else {
			return 0;
		}
	}
	
	private int totalClassifications() {
		int sum = 0;
		for (Integer a : categoryUsage.values()) {
			sum += a;
		}
		return sum;
	}
	
	private Set<NaiveBayesSpamClassification> categories() {
		return categoryUsage.keySet();
	}
	
	private double featureClassificationProbability(String feature, NaiveBayesSpamClassification classification) {
		if (featureCategory.containsKey(feature)) {
			return (double) featureClassificationCount(feature, classification) / classificationCount(classification);
		} else {
			return 0;
		}
	}
	
	private double weightedFeatureClassificationProbability(String feature, NaiveBayesSpamClassification classification) {
		double basicProbability = featureClassificationProbability(feature, classification);
		
		int totalClassifications = 0;
		for (NaiveBayesSpamClassification c : categories()) {
			totalClassifications += featureClassificationCount(feature, c);
		}
		
		double weightedProbability = ((ASSUMED_PROBABILITY_WEIGHT * ASSUMED_PROBABILITY) + (totalClassifications * basicProbability)) / (ASSUMED_PROBABILITY_WEIGHT + totalClassifications);
		return weightedProbability;
	}
	
	private double documentProbability(String document, NaiveBayesSpamClassification classification) {
		String[] features = getFeatures(document);
		double propability = 1;
		
		for (String f : features) {
			propability *= weightedFeatureClassificationProbability(f, classification);
		}
		
		return propability;
	}
	
	private double probabilityOfCategory(String document, NaiveBayesSpamClassification classification) {
		double categoryProbability = classificationCount(classification) / totalClassifications();
		double documentProbability = documentProbability(document, classification);
		
		return categoryProbability * documentProbability;
	}
	
	private void train(String textBody, NaiveBayesSpamClassification classification) {
		String[] features = getFeatures(textBody);
		for (String word : features) {
			increaseFeatureFrequencyClassificationCount(word, classification);
		}
		
		increaseCategoryCount(classification);
	}
}