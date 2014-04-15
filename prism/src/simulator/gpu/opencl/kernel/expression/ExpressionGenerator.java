//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel.expression;

import java.util.HashMap;
import java.util.Map;

import parser.ast.ExpressionFunc;
import prism.Pair;
import prism.Preconditions;
import simulator.gpu.automaton.PrismVariable;
import simulator.gpu.automaton.update.Action;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.opencl.kernel.memory.ArrayType;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.ExpressionValue;
import simulator.gpu.opencl.kernel.memory.PointerType;
import simulator.gpu.opencl.kernel.memory.StructureType;

/**
 * @author mcopik
 *
 */
public class ExpressionGenerator
{
	static public CLVariable accessStructureField(CLVariable structure, String fieldName)
	{
		Preconditions.checkCondition(structure.varType instanceof StructureType, "Can access field only in structure variable");
		StructureType type = (StructureType) structure.varType;
		CLVariable field = null;
		for (CLVariable var : type.getFields()) {
			if (var.varName.equals(fieldName)) {
				field = var;
				break;
			}
		}
		return field != null ? new CLVariable(field.varType, structure.varName + "." + field.varName) : null;
	}

	static public <T> Expression fromString(T object)
	{
		Preconditions.checkNotNull(object, "fromString() called on null reference!");
		return new Expression(object.toString());
	}

	static public Expression createAssignment(CLVariable dest, Expression expr)
	{
		Expression ret = createBasicExpression(dest.getSource(), Operator.AS, expr);
		//ret.exprString += ";";
		return ret;
	}

	static public Expression createAssignment(CLVariable dest, CLVariable source)
	{
		return createAssignment(dest, source.getName());
	}

	public enum Operator {
		GT, LT, GE, LE, EQ, NE, LAND, LOR, AS, ADD, SUB, MUL, DIV, ADD_AUGM, SUB_AUGM, MUL_AUGM, DIV_AUGM, LAND_AUGM
	};

	private static final Map<Operator, String> operatorsSource;
	static {
		operatorsSource = new HashMap<>();
		operatorsSource.put(Operator.GT, ">");
		operatorsSource.put(Operator.LT, "<");
		operatorsSource.put(Operator.GE, ">=");
		operatorsSource.put(Operator.LE, "<=");
		operatorsSource.put(Operator.EQ, "==");
		operatorsSource.put(Operator.LAND, "&&");
		operatorsSource.put(Operator.LOR, "||");
		operatorsSource.put(Operator.NE, "!=");
		operatorsSource.put(Operator.AS, "=");
		operatorsSource.put(Operator.ADD, "+");
		operatorsSource.put(Operator.SUB, "-");
		operatorsSource.put(Operator.MUL, "*");
		operatorsSource.put(Operator.DIV, "/");
		operatorsSource.put(Operator.ADD_AUGM, "+=");
		operatorsSource.put(Operator.SUB_AUGM, "-=");
		operatorsSource.put(Operator.MUL_AUGM, "*=");
		operatorsSource.put(Operator.DIV_AUGM, "/=");
		operatorsSource.put(Operator.LAND_AUGM, "&=");
	}

	//	static public Expression createBasicExpression(CLValue var, Operator operator, CLValue var2)
	//	{
	//		return ExpressionGenerator.createBasicExpression(var, operator, var2.getSource());
	//	}
	//
	//	static public Expression createBasicExpression(Expression var, Operator operator, CLValue expr)
	//	{
	//		return new Expression(String.format("%s %s %s", var, operatorsSource.get(operator), expr.getSource()));
	//	}
	//
	//	static public Expression createBasicExpression(CLValue var, Operator operator, String expr)
	//	{
	//		return new Expression(String.format("%s %s %s", var.getSource(), operatorsSource.get(operator), expr));
	//	}
	//
	//	static public Expression createBasicExpression(CLValue var, Operator operator, Expression expr)
	//	{
	//		return ExpressionGenerator.createBasicExpression(var, operator, expr.getSource());
	//	}
	static public Expression createBasicExpression(Expression expr1, Operator operator, Expression expr2)
	{
		return new Expression(String.format("%s %s %s", expr1, operatorsSource.get(operator), expr2));
	}

	static public Expression createNegation(Expression var)
	{
		var.exprString = "!(" + var.exprString + ")";
		return var;
	}

	static public Expression createConditionalAssignment(Expression condition, String first, String second)
	{
		return new Expression(String.format("%s ? %s : %s", condition.getSource(), first, second));
	}

	static public CLVariable accessArrayElement(CLVariable var, Expression indice)
	{
		Preconditions.checkCondition(var.varType.isArray(), String.format("Var %s is not an array!", var.varName));
		if (var.varType instanceof ArrayType) {
			return new CLVariable(((ArrayType) var.varType).getInternalType(),
			//varName[indice]
					String.format("%s[%s]", var.varName, indice));
		} else {
			return new CLVariable(((PointerType) var.varType).getInternalType(),
			//varName[indice]
					String.format("%s[%s]", var.varName, indice));
		}
	}

	static public Expression addParentheses(Expression expr)
	{
		expr.exprString = String.format("(%s)", expr.exprString);
		return expr;
	}

	static public Expression addComma(Expression expr)
	{
		expr.exprString = expr.exprString + ";";
		return expr;
	}

	static public Expression postIncrement(CLVariable var)
	{
		return new Expression(var.varName + "++");
	}

	static public CLValue assignGlobalID()
	{
		return new ExpressionValue(new Expression("get_global_id(0)"));
	}

	static public Expression functionCall(String functionName, Expression... args)
	{
		StringBuilder builder = new StringBuilder(functionName);
		builder.append("(");
		for (Expression arg : args) {
			builder.append(arg.getSource()).append(",");
		}
		builder.deleteCharAt(builder.length() - 1);
		builder.append(")");
		return new Expression(builder.toString());
	}

	static public Expression convertPrismAction(Action action)
	{
		StringBuilder builder = new StringBuilder();
		for (Pair<PrismVariable, parser.ast.Expression> expr : action.expressions) {
			//			builder.append(expr.first.name).append(" = ").append(expr.second.toString()).append(";");
			//			builder.append("\n");
			builder.append(expr.first.name).append(" = ");
			builder.append(convertUpdate(expr.second, null));
			builder.append(";\n");
		}
		return new Expression(builder.toString());
	}

	static public KernelComponent convertPrismAction(Action action, CLVariable changeFlag, CLVariable oldValue)
	{
		ExpressionList list = new ExpressionList();
		StringBuilder builder = new StringBuilder();
		for (Pair<PrismVariable, parser.ast.Expression> expr : action.expressions) {
			list.addExpression(createAssignment(oldValue, new Expression(expr.first.name)));
			Expression mainAssignment = new Expression(String.format("%s = %s", expr.first.name, convertUpdate(expr.second, null)));
			addParentheses(mainAssignment);
			list.addExpression(createBasicExpression(changeFlag.getSource(), Operator.LAND_AUGM, createConditionalAssignment(
			//destination == new_value
					createBasicExpression(mainAssignment, Operator.EQ, oldValue.getSource()), "true", "false")));

			//			builder.append(expr.first.name).append(" = ");
			//			builder.append(createAssignment(oldValue, new Expression(convertUpdate(expr.second, null)))).append("\n");
			//			builder.append(createAssignment(changeFlag, createConditionalAssignment(
			//			//destination == new_value
			//					createBasicExpression(new Expression(expr.first.name), Operator.EQ, oldValue.getSource()), "true", "false"))).append("\n");
			//			builder.append(expr.first.name).append(" = ").append(oldValue.getSource()).append(";").append("\n");
			//			builder.append("\n");
		}
		return list;
	}

	static public Expression convertPrismAction(Action action, Map<String, String> translations)
	{
		StringBuilder builder = new StringBuilder();
		for (Pair<PrismVariable, parser.ast.Expression> expr : action.expressions) {

			builder.append(expr.first.name).append(" = ");
			builder.append(convertUpdate(expr.second, translations));
			builder.append(";\n");
		}
		return new Expression(builder.toString());
	}

	static public KernelComponent convertPrismAction(Action action, Map<String, String> translations, CLVariable changeFlag, CLVariable oldValue)
	{
		ExpressionList list = new ExpressionList();
		StringBuilder builder = new StringBuilder();
		for (Pair<PrismVariable, parser.ast.Expression> expr : action.expressions) {

			list.addExpression(createAssignment(oldValue, new Expression(expr.first.name)));
			Expression mainAssignment = new Expression(String.format("%s = %s", expr.first.name, convertUpdate(expr.second, translations)));
			addParentheses(mainAssignment);
			list.addExpression(createBasicExpression(changeFlag.getSource(), Operator.LAND_AUGM, createConditionalAssignment(
			//destination == new_value
					createBasicExpression(mainAssignment, Operator.EQ, oldValue.getSource()), "true", "false")));
			//			list.addExpression(createAssignment(changeFlag, createConditionalAssignment(
			//			//destination == new_value
			//					createBasicExpression(mainAssignment, Operator.EQ, oldValue.getSource()), "true", "false")));
			//builder.append(expr.first.name).append(" = ").append(oldValue.getSource()).append(";").append("\n");
			//builder.append("\n");
		}
		return list;
	}

	static private String convertUpdate(parser.ast.Expression expr, Map<String, String> translations)
	{
		StringBuilder assignment = new StringBuilder();
		convertFunc(assignment, translations, expr);
		convertEquality(assignment);
		builderReplace(assignment, "|", "||");
		builderReplace(assignment, "&", "&&");
		return assignment.toString();
	}

	static private void convertFunc(StringBuilder builder, Map<String, String> translations, parser.ast.Expression expr)
	{
		if (expr instanceof ExpressionFunc) {
			ExpressionFunc func = (ExpressionFunc) expr;
			builder.append(func.getName()).append('(');
			for (int i = 0; i < func.getNumOperands(); ++i) {
				if (func.getOperand(i) instanceof ExpressionFunc) {
					builder.append("(float)");
					convertFunc(builder, translations, func.getOperand(i));
				} else {
					if (translations != null) {
						String newExpr = convertActionWithSV(translations, func.getOperand(i).toString());
						//no change? cast to float for overloading functions e.g. min to (float,float), not (float,int)l
						if (newExpr.equals(func.getOperand(i).toString())) {
							builder.append("((float)").append(newExpr).append(")");
						} else {
							builder.append(newExpr);
						}
					} else {
						builder.append(func.getOperand(i).toString());
					}
					if (i != func.getNumOperands() - 1) {
						builder.append(',');
					}
				}
			}
			builder.append(")");
		} else {
			if (translations != null) {
				builder.append(convertActionWithSV(translations, expr.toString()));
			} else {
				builder.append(expr.toString());
			}
		}
	}

	//	static private String convertEquality(String expr)
	//	{
	//		StringBuilder builder = new StringBuilder(expr);
	//		int index = 0;
	//		while ((index = builder.indexOf("=", index)) != -1) {
	//			if (index == 0 || (builder.charAt(index - 1) != '!' && builder.charAt(index - 1) != '>' && builder.charAt(index - 1) != '<')) {
	//				builder.replace(index, index + 1, "==");
	//				index += 2;
	//			} else {
	//				index += 1;
	//			}
	//		}
	//		return builder.toString();
	//	}

	static private void convertEquality(StringBuilder builder)
	{
		int index = 0;
		while ((index = builder.indexOf("=", index)) != -1) {
			if (index == 0 || (builder.charAt(index - 1) != '!' && builder.charAt(index - 1) != '>' && builder.charAt(index - 1) != '<')) {
				builder.replace(index, index + 1, "==");
				index += 2;
			} else {
				index += 1;
			}
		}
	}

	static private String convertActionWithSV(Map<String, String> translations, String action)
	{
		StringBuilder builder = new StringBuilder(action);
		for (Map.Entry<String, String> entry : translations.entrySet()) {
			//while ((index = builder.indexOf(entry.getKey(), index)) != -1) {

			builderReplaceMostCommon(builder, entry.getKey(), String.format("((float)%s)", entry.getValue()));

			//builder.replace(index, index + entry.getKey().length(), String.format("((float)%s)", entry.getValue()));
			//index += entry.getValue().length() + 9;
			//}
		}
		return builder.toString();
	}

	static public String convertPrismRate(PrismVariable[] stateVector, Rate rate)
	{
		StringBuilder builder = new StringBuilder(rate.toString());
		for (int i = 0; i < stateVector.length; ++i) {
			//			while ((index = builder.indexOf(stateVector[i].name, index)) != -1) {
			//				builder.replace(index, index + stateVector[i].name.length(), String.format("((float)%s)", stateVector[i].name));
			//				index += stateVector[i].name.length() + 9;
			//			}
			builderReplaceMostCommon(builder, stateVector[i].name, String.format("((float)%s)", stateVector[i].name));
		}
		return builder.toString();
	}

	static public Expression convertPrismProperty(PrismVariable[] vars, String expr)
	{
		StringBuilder builder = new StringBuilder(expr);
		for (int i = 0; i < vars.length; ++i) {
			builderReplaceMostCommon(builder, vars[i].name, String.format("((float)%s)", vars[i].name));
		}
		convertEquality(builder);
		builderReplace(builder, "|", "||");
		builderReplace(builder, "&", "&&");
		//String newExpr = expr.replace("=", "==").replace("&", "&&").replace("|", "||");
		//		if (expr.charAt(0) == ('!')) {
		//			return new Expression(String.format("!(%s)", newExpr.substring(1)));
		//		} else {
		//			return new Expression(newExpr);
		//		}
		//return new Expression(newExpr);
		return new Expression(builder.toString());
	}

	static public String convertPrismGuard(PrismVariable[] vars, String expr)
	{
		StringBuilder builder = new StringBuilder(expr);
		for (int i = 0; i < vars.length; ++i) {
			builderReplaceMostCommon(builder, vars[i].name, String.format("((float)%s)", vars[i].name));
		}
		convertEquality(builder);
		builderReplace(builder, "|", "||");
		builderReplace(builder, "&", "&&");
		if (builder.charAt(0) == ('!')) {
			return String.format("!(%s)", builder.substring(1));
		} else {
			return builder.toString();
		}
	}

	static private void builderReplace(StringBuilder builder, String first, String second)
	{
		int index = 0;
		while ((index = builder.indexOf(first, index)) != -1) {
			builder.replace(index, index + first.length(), second);
			index += second.length();
		}
	}

	static private void builderReplaceMostCommon(StringBuilder builder, String first, String second)
	{
		int index = 0;
		while ((index = builder.indexOf(first, index)) != -1) {
			//check whether it is a prefix
			if (builder.length() != index + first.length()
					&& (Character.isAlphabetic(builder.charAt(index + first.length())) || Character.isDigit(builder.charAt(index + first.length())))) {
				index += first.length();
			}
			//check if it is a suffix
			else if (index != 0 && (Character.isAlphabetic(builder.charAt(index - 1)) || Character.isDigit(builder.charAt(index - 1)))) {
				index += first.length();
			} else {
				builder.replace(index, index + first.length(), second);
				index += second.length();
			}
			//for safety
			index = Math.min(index, builder.length() - 1);
		}
	}
}